"""Readout transfer routing, failure and saved-prefix regression checks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import requests

from scripts import stage_population_readout_inputs as transfer


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.auth = self.root / "auth.json"
        self.auth.write_text(json.dumps({"jwt_token": "fixture-jwt", "cookies": {"fixture": "cookie"}}))
        self.environment = patch.dict(os.environ, {"MERLIN_AUTH_FILE": str(self.auth),
                                                   "MERLIN_DATA_CONTROL_PLANE": "cn-seed"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.session = MagicMock()
        self.session.headers = {}
        self.session.cookies = MagicMock()
        self.helper = SimpleNamespace(http_session=lambda: self.session)
        self.response = MagicMock()
        self.response.json.return_value = {"ResponseBase": {"code": 0},
                                            "data": {"size_bytes": 4, "hdfs_path": "fixture/result"}}
        self.session.post.return_value = self.response

    def test_data_plane_selects_matching_endpoint_and_auth(self):
        for plane, host in (("cn-seed", "cluster-seed.example.invalid"), ("cn", "cluster-api.example.invalid")):
            with self.subTest(plane=plane), patch.dict(os.environ, {"MERLIN_DATA_CONTROL_PLANE": plane}):
                self.assertEqual(transfer.multipart_chunk(b"data", "part", self.helper)["size_bytes"], 4)
                self.assertEqual(self.session.post.call_args.args[0],
                                 "https://" + host + "/api/v4/seed/data/datacard-upload-files/create")
                self.assertEqual(self.session.headers["x-jwt-token"], "fixture-jwt")
                self.session.cookies.clear.assert_called()
                self.session.cookies.update.assert_called_with({"fixture": "cookie"})

    def test_default_auth_file_belongs_to_data_plane(self):
        auth = self.root / ".merlin-cli/auth/cn-seed.json"
        auth.parent.mkdir(parents=True)
        auth.write_text(self.auth.read_text())
        with patch.dict(os.environ, {"MERLIN_AUTH_FILE": ""}), patch.object(Path, "home", return_value=self.root):
            transfer.multipart_chunk(b"data", "part", self.helper)
        self.assertEqual(self.session.headers["x-jwt-token"], "fixture-jwt")

    def test_authentication_failure_stops_without_retry(self):
        error = requests.HTTPError("expired", response=SimpleNamespace(status_code=401))
        self.response.raise_for_status.side_effect = error
        with patch.object(transfer.time, "sleep") as sleep, self.assertRaises(requests.HTTPError):
            transfer.multipart_chunk(b"data", "part", self.helper)
        self.assertEqual(self.session.post.call_count, 1)
        sleep.assert_not_called()

    def test_transient_failure_has_three_attempt_limit(self):
        self.session.post.side_effect = requests.ConnectionError("EOF")
        with patch.object(transfer.time, "sleep") as sleep, self.assertRaises(requests.ConnectionError):
            transfer.multipart_chunk(b"data", "part", self.helper)
        self.assertEqual(self.session.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_partial_archive_reuses_verified_mixed_length_prefix(self):
        archive, manifest = self.root / "inputs.tar", self.root / "upload.json"
        archive.write_bytes(b"abcdefghij")
        old = [{"index": 0, "bytes": 2, "sha256": hashlib.sha256(b"ab").hexdigest(), "hdfs_path": "old/0"},
               {"index": 1, "bytes": 3, "sha256": hashlib.sha256(b"cde").hexdigest(), "hdfs_path": "old/1"}]
        manifest.write_text(json.dumps({"file": str(archive.resolve()), "bytes": 10, "chunks": old,
                                        "complete": False}))
        def save(path, value):
            path.write_text(json.dumps(value))
        helper = SimpleNamespace(HTTP_LOCAL=SimpleNamespace(), atomic_json=save)
        uploaded = []
        def upload(data, name, unused):
            uploaded.append(data)
            return {"size_bytes": len(data), "hdfs_path": "new/" + name}
        with patch.object(transfer, "multipart_chunk", side_effect=upload):
            result = transfer.upload_archive(archive, manifest, 4, helper)
        self.assertEqual(uploaded, [b"fghi", b"j"])
        self.assertEqual(result["chunks"][:2], old)
        self.assertTrue(result["complete"])
        self.assertEqual(result["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
