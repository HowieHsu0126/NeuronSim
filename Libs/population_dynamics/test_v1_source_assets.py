"""Source identity, incomplete download and bounded metadata transfer checks."""
import base64
import hashlib
import json
import io
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests
from urllib3.response import HTTPResponse

from . import v1_source_assets as assets
from .v1_source_assets import download_resolved, fetch_child, folder_urls, published_child


class SourceAssetTests(unittest.TestCase):
    def test_only_observed_folder_urls_are_used(self):
        url = "https://www.dropbox.com/scl/fo/folder/asset/network/v1_nodes.h5?rlkey=key&dl=0"
        key = base64.b64encode(b"SharedContentLinkFolderEntriesPropsPrefetch").decode()
        value = base64.b64encode(b"\0" + url.encode() + b"\0").decode()
        page = f'registerStreamedPrefetch("{key}","{value}")'
        self.assertEqual(folder_urls(page), [url])
        self.assertEqual(folder_urls(page.replace('"', '\\"')), [url])
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index.html"
            index.write_text(page)
            self.assertEqual(published_child(index, "v1_nodes.h5"), url)
            with self.assertRaisesRegex(ValueError, "ONE_PUBLISHED"):
                published_child(index, "invented.h5")

    def test_same_named_child_excludes_receipt_bound_parent(self):
        parent = "https://www.dropbox.com/scl/fo/folder/parent/lgnmodel?rlkey=key&dl=0"
        child = "https://www.dropbox.com/scl/fo/folder/child/lgnmodel/lgnmodel?rlkey=key&dl=0"
        key = base64.b64encode(b"SharedContentLinkFolderEntriesPropsPrefetch").decode()
        def page(urls):
            value = base64.b64encode("\0".join(urls).encode()).decode()
            return f'registerStreamedPrefetch("{key}","{value}")'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "asset_index.html"
            index.write_text(page([parent, child]))
            with self.assertRaisesRegex(ValueError, "ONE_PUBLISHED"):
                published_child(index, "lgnmodel")
            (root / "source.json").write_text(json.dumps(
                dict(artifact=index.name, published_url=parent)))
            self.assertEqual(published_child(index, "lgnmodel"), child)
            index.write_text(page([parent]))
            with self.assertRaisesRegex(ValueError, "ONE_PUBLISHED"):
                published_child(index, "lgnmodel")

    def test_frozen_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "source.json").write_text('{"preserve": true}')
            with patch("requests.get") as request:
                with self.assertRaises(FileExistsError):
                    fetch_child(output / "missing.html", "asset", output)
                request.assert_not_called()
            self.assertEqual((output / "source.json").read_text(), '{"preserve": true}')

    def _download(self, temporary, *, payload=b"abcd", expected_size=4, etag="original", expected_sha256=None):
        root = Path(temporary)
        source = dict(artifact="nodes.h5", expected_bytes=expected_size, etag="original",
                      resolved_url="https://example.invalid/expired", published_url="https://example.invalid/shared?rlkey=key&dl=0")
        if expected_sha256:
            source["expected_sha256"] = expected_sha256
        receipt = root / "download.json"
        receipt.write_text(json.dumps(source))
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"ETag": etag, "Content-Range": f"bytes 0-{expected_size-1}/{expected_size}"}
        response.status_code = 206
        response.url = "https://example.invalid/current"
        response.iter_content.return_value = [payload]
        session = MagicMock()
        session.get.return_value = response
        with patch("requests.Session", return_value=session):
            result = download_resolved(receipt, root / "asset", metadata_transfer=True)
        return result, session

    def test_exact_transfer_and_fresh_public_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, session = self._download(temporary, expected_sha256=hashlib.sha256(b"abcd").hexdigest())
            self.assertEqual(result["bytes"], 4)
            self.assertTrue(result["asset_body_acquired"])
            self.assertFalse(result["complete_model_acquired"])
            self.assertEqual((Path(temporary) / "asset/nodes.h5").read_bytes(), b"abcd")
            self.assertEqual(session.get.call_args.args[0], "https://example.invalid/shared?rlkey=key&dl=1")

    def test_truncation_and_changed_etag_remain_failures(self):
        for payload, etag, error in [(b"abc", "original", "INCOMPLETE"), (b"abcd", "changed", "ETAG_CHANGED")]:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, error):
                    self._download(temporary, payload=payload, etag=etag)
                output = Path(temporary) / "asset"
                self.assertFalse((output / "nodes.h5").exists())
                self.assertEqual(json.loads((output / "source.json").read_text())["status"], "FAILED")

    def test_bulk_bytes_cannot_use_metadata_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "SMALL_FILE_LIMIT"):
                self._download(temporary, expected_size=16 * 1024 * 1024 + 1)
            self.assertFalse((Path(temporary) / "asset").exists())

    def test_verified_timeout_prefix_resumes_at_exact_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = b"a" * (512 * 1024)
            source = dict(artifact="nodes.h5", expected_bytes=len(prefix)+4, etag="version",
                          published_url="https://example.invalid/shared?rlkey=key&dl=0",
                          resolved_url="https://example.invalid/expired",
                          expected_sha256=hashlib.sha256(prefix + b"tail").hexdigest())
            receipt = root / "download.json"
            receipt.write_text(json.dumps(source))
            prior = root / "prior"
            prior.mkdir()
            (prior / "nodes.h5.part").write_bytes(prefix)
            (prior / "source.json").write_text(json.dumps(dict(source=source, error="V1_METADATA_TRANSFER_TIME_LIMIT", received_bytes=len(prefix))))
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 206
            response.url = "https://example.invalid/current"
            response.headers = {"ETag": "version", "Content-Range": f"bytes {len(prefix)}-{len(prefix)+3}/{len(prefix)+4}"}
            response.iter_content.return_value = [b"tail"]
            session = MagicMock()
            session.get.return_value = response
            with patch("requests.Session", return_value=session):
                result = download_resolved(receipt, root / "resumed", metadata_transfer=True, resume_metadata=prior)
            self.assertEqual(result["bytes"], len(prefix)+4)
            self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], f"bytes={len(prefix)}-{len(prefix)+3}")
            self.assertEqual((root / "resumed/nodes.h5").read_bytes(), prefix + b"tail")
            self.assertEqual((prior / "nodes.h5.part").read_bytes(), prefix)


class BulkTransportTests(unittest.TestCase):
    """Exercise actual file/receipt behavior with bounded mocked source responses."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for context in (
            patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0", "SLURMD_NODENAME": "gpu03"}),
            patch.object(assets.socket, "gethostname", return_value="gpu03"),
            patch.object(assets.subprocess, "run", return_value=MagicMock(stdout="xfs\n")),
            patch.object(assets.shutil, "disk_usage", return_value=MagicMock(free=2**40)),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.source = dict(artifact="edges.h5", expected_bytes=8, etag='"source-version"',
                           resolved_url="https://example.invalid/stale-cdn",
                           published_url="https://example.invalid/shared?rlkey=key&dl=0",
                           expected_sha256=hashlib.sha256(b"abcdefgh").hexdigest())
        self.receipt = self.root / "download.json"
        self._save_source()

    def _save_source(self):
        self.receipt.write_text(json.dumps(self.source))

    def _response(self, payload, start, stop, *, status=206, etag=None, total=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.headers = {"ETag": self.source["etag"] if etag is None else etag,
                            "Content-Range": f"bytes {start}-{stop}/{self.source['expected_bytes'] if total is None else total}"}
        response.url = "https://example.invalid/fresh-cdn"
        response.iter_content.return_value = [payload]
        return response

    def _run(self, responses, output="asset", **options):
        session = MagicMock()
        session.get.side_effect = responses
        with patch.object(assets.requests, "Session", return_value=session):
            report = download_resolved(self.receipt, self.root / output, **options)
        return report, session

    def _partial(self):
        report, _ = self._run([self._response(b"abcd", 0, 3)], output="prior",
                              asset_trust_env=True, range_bytes=4, max_transfer_bytes=4)
        self.assertEqual(report["status"], "PARTIAL")
        return self.root / "prior"

    def test_environment_bulk_uses_exact_ranges_without_default_total_timeout(self):
        with patch.object(assets.time, "monotonic") as clock:
            report, session = self._run([self._response(b"abcd", 0, 3), self._response(b"efgh", 4, 7)],
                                        asset_trust_env=True, range_bytes=4)
        clock.assert_not_called()
        self.assertTrue(session.trust_env)
        self.assertEqual(report["attempt_seconds"], None)
        self.assertEqual(report["storage"]["required_bytes"], 8)
        self.assertEqual((self.root / "asset/edges.h5").read_bytes(), b"abcdefgh")
        self.assertEqual([call.kwargs["headers"]["Range"] for call in session.get.call_args_list],
                         ["bytes=0-3", "bytes=4-7"])
        for call in session.get.call_args_list:
            self.assertEqual(call.args[0], self.source["published_url"].replace("&dl=0", "&dl=1"))
            self.assertEqual(call.kwargs["headers"]["If-Match"], self.source["etag"])

    def test_default_direct_stream_remains_supported(self):
        report, session = self._run([self._response(b"abcdefgh", 0, 7, status=200)])
        self.assertFalse(session.trust_env)
        self.assertFalse(report["ranged_transfer"])
        self.assertEqual(session.get.call_args.args[0], self.source["resolved_url"])
        self.assertNotIn("Range", session.get.call_args.kwargs["headers"])

    def test_range_probe_of_large_asset_is_bounded_and_does_not_claim_intake(self):
        self.source["expected_bytes"] = 20 * 1024 * 1024
        self._save_source()
        with patch.object(assets.shutil, "disk_usage", return_value=MagicMock(free=8)):
            report, session = self._run([self._response(b"abcd", 0, 3), self._response(b"efgh", 4, 7)],
                                        asset_trust_env=True, range_bytes=4, probe_bytes=8)
        self.assertEqual(report["status"], "SOURCE_RANGE_PROBED")
        self.assertEqual(report["probed_bytes"], 8)
        self.assertFalse(report["asset_body_acquired"])
        self.assertFalse((self.root / "asset/edges.h5").exists())
        self.assertEqual((self.root / "asset/edges.h5.part").read_bytes(), b"abcdefgh")
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(report["storage"]["required_bytes"], 8)

    def test_byte_limited_attempt_resumes_unaligned_verified_prefix(self):
        prior_report, _ = self._run([self._response(b"abcd", 0, 3), self._response(b"e", 4, 4)],
                                    output="prior", range_bytes=4, max_transfer_bytes=5)
        self.assertEqual(prior_report["stop_reason"], "V1_ASSET_TRANSFER_BYTE_LIMIT")
        self.assertEqual(prior_report["verified_prefix_bytes"], 5)
        report, session = self._run([self._response(b"fgh", 5, 7)], resume_from=self.root / "prior", range_bytes=4)
        self.assertEqual(report["resumed_prefix_bytes"], 5)
        self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=5-7")
        self.assertEqual((self.root / "asset/edges.h5").read_bytes(), b"abcdefgh")
        self.assertEqual((self.root / "prior/edges.h5.part").read_bytes(), b"abcde")

    def test_incomplete_range_tail_is_preserved_but_not_reused(self):
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            self._run([self._response(b"abcd", 0, 3), self._response(b"ef", 4, 7)], output="prior", range_bytes=4)
        prior = json.loads((self.root / "prior/source.json").read_text())
        self.assertEqual((prior["received_bytes"], prior["verified_prefix_bytes"]), (6, 4))
        self.assertEqual((self.root / "prior/edges.h5.part").read_bytes(), b"abcdef")
        report, session = self._run([self._response(b"efgh", 4, 7)], resume_from=self.root / "prior", range_bytes=4)
        self.assertTrue(report["asset_body_acquired"])
        self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=4-7")

    def test_archived_prefix_is_staged_locally_before_network_resume(self):
        self.source["expected_md5"] = hashlib.md5(b"abcdefgh").hexdigest()
        self._save_source()
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            self._run([self._response(b"abcd", 0, 3), self._response(b"ef", 4, 7)],
                      output="archive", range_bytes=4)
        archive = self.root / "archive"
        original = {name: (archive / name).read_bytes()
                    for name in ("source.json", "edges.h5.part")}
        output = self.root / "asset"

        def filesystem(command, **kwargs):
            path = Path(command[command.index("-T") + 1])
            return MagicMock(stdout="nfs4\n" if path == archive else "xfs\n")

        def resumed_response(*args, **kwargs):
            # The read-only archive has already been copied into local working
            # storage; the incomplete range's tail must not enter the new asset.
            self.assertEqual(kwargs["headers"]["Range"], "bytes=4-7")
            self.assertTrue((output / "edges.h5.part").exists())
            return self._response(b"efgh", 4, 7)

        session = MagicMock()
        session.get.side_effect = resumed_response
        with patch.object(assets.subprocess, "run", side_effect=filesystem), \
                patch.object(assets.requests, "Session", return_value=session):
            report = download_resolved(self.receipt, output, resume_from=archive, range_bytes=4)
        self.assertEqual(report["resumed_prefix_bytes"], 4)
        self.assertTrue(report["checksum_verification"]["md5"]["verified"])
        self.assertEqual((output / "edges.h5").read_bytes(), b"abcdefgh")
        self.assertEqual({name: (archive / name).read_bytes() for name in original}, original)
        session.get.assert_called_once()

    def test_resume_rejects_modified_prefix_before_network(self):
        prior = self._partial()
        (prior / "edges.h5.part").write_bytes(b"xbcd")
        with patch.object(assets.requests, "Session") as constructor:
            with self.assertRaisesRegex(ValueError, "PREFIX_HASH_CHANGED"):
                download_resolved(self.receipt, self.root / "asset", resume_from=prior)
        constructor.return_value.get.assert_not_called()
        self.assertEqual((prior / "edges.h5.part").read_bytes(), b"xbcd")

    def test_resume_rejects_changed_source_identity_before_network(self):
        prior = self._partial()
        self.source["etag"] = '"replacement"'
        self._save_source()
        with patch.object(assets.requests, "Session") as constructor:
            with self.assertRaisesRegex(ValueError, "VERIFIED_ASSET_PREFIX_REQUIRED"):
                download_resolved(self.receipt, self.root / "asset", resume_from=prior)
        constructor.return_value.get.assert_not_called()

    def test_bad_range_or_identity_never_becomes_an_asset(self):
        cases = [
            (self._response(b"abcdefgh", 0, 7, status=200), "RANGE_MISMATCH"),
            (self._response(b"abcdefgh", 0, 7, total=9), "RANGE_MISMATCH"),
            (self._response(b"abcdefgh", 0, 7, etag='"replacement"'), "ETAG_CHANGED"),
            (self._response(b"abcdefghi", 0, 7), "SIZE_EXCEEDED"),
        ]
        for number, (response, reason) in enumerate(cases):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self._run([response], output=str(number), range_bytes=8)
            output = self.root / str(number)
            self.assertFalse((output / "edges.h5").exists())
            self.assertEqual(json.loads((output / "source.json").read_text())["status"], "FAILED")

    def test_strong_etag_is_required_before_ranged_requests(self):
        self.source.pop("expected_sha256")
        for etag in (None, "", 'W/"weak"'):
            with self.subTest(etag=etag):
                self.source["etag"] = etag
                self._save_source()
                with patch.object(assets.requests, "Session") as constructor:
                    with self.assertRaisesRegex(ValueError, "STRONG_ETAG"):
                        download_resolved(self.receipt, self.root / "asset", asset_trust_env=True)
                constructor.assert_not_called()


    def _checksum_source(self, algorithm="md5"):
        self.source["etag"] = None
        self.source.pop("expected_sha256", None)
        self.source["expected_" + algorithm] = hashlib.new(algorithm, b"abcdefgh").hexdigest()
        self._save_source()

    def _without_etag_response(self, payload, start, stop):
        response = self._response(payload, start, stop)
        response.headers.pop("ETag")
        response.headers["Content-Type"] = "application/octet-stream"
        return response

    def test_no_etag_full_checksum_allows_exact_range_acquisition(self):
        for algorithm in ("md5", "sha256"):
            with self.subTest(algorithm=algorithm):
                self.source.pop("expected_md5", None)
                self._checksum_source(algorithm)
                expected = self.source["expected_" + algorithm]
                self.source["expected_" + algorithm] = expected.upper()
                self._save_source()
                report, session = self._run([
                    self._without_etag_response(b"abcd", 0, 3),
                    self._without_etag_response(b"efgh", 4, 7),
                ], output=algorithm, range_bytes=4)
                self.assertTrue(report["asset_body_acquired"])
                self.assertEqual(report["range_identity"], "source_checksum")
                self.assertEqual(report["checksum_verification"][algorithm],
                                 dict(expected=expected, actual=expected, verified=True))
                self.assertEqual((self.root / algorithm / "edges.h5").read_bytes(), b"abcdefgh")
                for call in session.get.call_args_list:
                    self.assertNotIn("If-Match", call.kwargs["headers"])

    def test_wrong_whole_file_md5_never_publishes_an_asset(self):
        self._checksum_source()
        self.source["expected_md5"] = "0" * 32
        self._save_source()
        with self.assertRaisesRegex(ValueError, "MD5_CHANGED"):
            self._run([self._without_etag_response(b"abcdefgh", 0, 7)], range_bytes=8)
        self.assertFalse((self.root / "asset/edges.h5").exists())
        report = json.loads((self.root / "asset/source.json").read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse(report["asset_body_acquired"])
        self.assertEqual(report["checksum_verification"]["md5"],
                         dict(expected="0" * 32, actual=hashlib.md5(b"abcdefgh").hexdigest(), verified=False))

    def test_malformed_expected_checksum_is_rejected_before_any_transfer(self):
        for key, value in (("expected_md5", None), ("expected_md5", ""),
                           ("expected_md5", "g" * 32), ("expected_md5", "0" * 31),
                           ("expected_sha256", "0" * 63)):
            with self.subTest(key=key, value=value):
                self.source[key] = value
                self._save_source()
                with patch.object(assets.requests, "Session") as constructor:
                    with self.assertRaisesRegex(ValueError, "V1_INVALID_EXPECTED_"):
                        download_resolved(self.receipt, self.root / "asset", range_bytes=8)
                constructor.assert_not_called()
                self.assertFalse((self.root / "asset").exists())
                self.source.pop(key)

    def test_checksum_bound_partial_resumes_and_verifies_whole_file(self):
        self._checksum_source()
        report, _ = self._run([self._without_etag_response(b"abcd", 0, 3)],
                              output="prior", range_bytes=4, max_transfer_bytes=4)
        self.assertEqual(report["verified_prefix_sha256"], hashlib.sha256(b"abcd").hexdigest())
        self.assertFalse(report["checksum_verification"]["md5"]["verified"])
        self.assertIsNone(report["checksum_verification"]["md5"]["actual"])
        report, session = self._run([self._without_etag_response(b"efgh", 4, 7)],
                                    range_bytes=4, resume_from=self.root / "prior")
        self.assertTrue(report["checksum_verification"]["md5"]["verified"])
        self.assertTrue(report["asset_body_acquired"])
        self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=4-7")
        self.assertEqual((self.root / "prior/edges.h5.part").read_bytes(), b"abcd")

    def test_checksum_bound_resume_rejects_changed_identity_or_prefix(self):
        self._checksum_source()
        prior = self._partial()
        original = self.source["expected_md5"]
        self.source["expected_md5"] = "0" * 32
        self._save_source()
        with patch.object(assets.requests, "Session") as constructor:
            with self.assertRaisesRegex(ValueError, "VERIFIED_ASSET_PREFIX_REQUIRED"):
                download_resolved(self.receipt, self.root / "changed_identity", resume_from=prior)
            constructor.return_value.get.assert_not_called()
        self.source["expected_md5"] = original
        self._save_source()
        (prior / "edges.h5.part").write_bytes(b"xbcd")
        with patch.object(assets.requests, "Session") as constructor:
            with self.assertRaisesRegex(ValueError, "PREFIX_HASH_CHANGED"):
                download_resolved(self.receipt, self.root / "changed_prefix", resume_from=prior)
            constructor.return_value.get.assert_not_called()

    def test_checksum_bound_probe_never_claims_whole_file_verification(self):
        self._checksum_source()
        report, _ = self._run([self._without_etag_response(b"abcd", 0, 3)], range_bytes=4, probe_bytes=4)
        self.assertEqual(report["status"], "SOURCE_RANGE_PROBED")
        self.assertFalse(report["asset_body_acquired"])
        self.assertFalse(report["checksum_verification"]["md5"]["verified"])
        self.assertIsNone(report["checksum_verification"]["md5"]["actual"])
        self.assertFalse((self.root / "asset/edges.h5").exists())

    def test_checksum_bound_ranges_still_require_exact_status_range_and_length(self):
        self._checksum_source()
        wrong_status = self._without_etag_response(b"abcdefgh", 0, 7)
        wrong_status.status_code = 200
        wrong_range = self._without_etag_response(b"abcdefgh", 0, 7)
        wrong_range.headers["Content-Range"] = "bytes 0-7/9"
        cases = [(wrong_status, "RANGE_MISMATCH"), (wrong_range, "RANGE_MISMATCH"),
                 (self._without_etag_response(b"abcdefg", 0, 7), "INCOMPLETE")]
        for number, (response, error) in enumerate(cases):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self._run([response], output=str(number), range_bytes=8)
            self.assertFalse((self.root / str(number) / "edges.h5").exists())

    def test_explicit_route_session_disables_environment_and_records_route(self):
        report, session = self._run([self._response(b"abcdefgh", 0, 7)],
                                    resolve_address="192.0.2.10", metadata_transfer=True)
        self.assertFalse(session.trust_env)
        self.assertEqual(report["asset_transport"], "explicit_address")
        self.assertEqual(report["resolved_address_route"],
                         dict(hostname="example.invalid", address="192.0.2.10", port=443,
                              tls_hostname="example.invalid", certificate_verification=True))
        self.assertEqual([call.args[0] for call in session.mount.call_args_list], ["https://", "http://"])
        self.assertIs(session.mount.call_args_list[0].args[1], session.mount.call_args_list[1].args[1])
        session.headers.__setitem__.assert_called_once_with("User-Agent", "NeuronSim-source-check/1.0")
        session.close.assert_called_once_with()
        self.assertEqual(session.get.call_args.args[0], self.source["published_url"].replace("&dl=0", "&dl=1"))

    def test_resolved_origin_routes_to_bound_cdn_without_publisher_redirect(self):
        self.source["resolved_url"] = "https://cdn.example.invalid/bound-file"
        self._save_source()
        report, session = self._run([self._response(b"abcdefgh", 0, 7)],
            resolve_address="192.0.2.10", range_bytes=8, use_resolved_url=True)
        self.assertEqual(session.get.call_args.args[0], self.source["resolved_url"])
        self.assertEqual(report["requested_url_source"], "resolved_url")
        self.assertEqual(report["resolved_address_route"]["hostname"], "cdn.example.invalid")
        self.assertEqual(report["resolved_address_route"]["tls_hostname"], "cdn.example.invalid")
        self.assertTrue(report["resolved_address_route"]["certificate_verification"])
        self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=0-7")
        self.assertEqual(session.get.call_args.kwargs["headers"]["If-Match"], self.source["etag"])

    def test_resolved_origin_still_rejects_changed_file_identity(self):
        response = self._response(b"abcdefgh", 0, 7)
        response.headers["ETag"] = "changed"
        with self.assertRaisesRegex(ValueError, "ETAG_CHANGED"):
            self._run([response], resolve_address="192.0.2.10", range_bytes=8, use_resolved_url=True)
        self.assertFalse((self.root / "asset/edges.h5").exists())

    def test_explicit_route_rejects_environment_conflict_before_transfer(self):
        with patch.object(assets.requests, "Session") as constructor:
            with self.assertRaisesRegex(ValueError, "CONFLICTS_WITH_ENVIRONMENT_TRANSPORT"):
                download_resolved(self.receipt, self.root / "asset", resolve_address="192.0.2.10", asset_trust_env=True)
        constructor.assert_not_called()
        self.assertFalse((self.root / "asset").exists())

    def test_cli_passes_explicit_route_and_requires_source_receipt(self):
        argv = ["v1_source_assets", "--source-receipt", str(self.receipt), "--output", str(self.root / "asset"),
                "--resolve-address", "192.0.2.10", "--range-bytes", "4", "--use-resolved-url"]
        with patch.object(sys, "argv", argv), patch.object(sys, "stdout", io.StringIO()):
            with patch.object(assets, "download_resolved", return_value={"status": "SOURCE_RANGE_PROBED"}) as run:
                assets.main()
        self.assertEqual(run.call_args.kwargs["resolve_address"], "192.0.2.10")
        self.assertTrue(run.call_args.kwargs["use_resolved_url"])
        argv = ["v1_source_assets", "--output", str(self.root / "asset"), "--resolve-address", "192.0.2.10"]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                assets.main()
        self.assertEqual(raised.exception.code, 2)

    def test_compute_allocation_is_required_even_with_proxy(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            with self.assertRaisesRegex(ValueError, "COMPUTE_ALLOCATION"):
                self._run([], asset_trust_env=True)
        with patch.object(assets.socket, "gethostname", return_value="mgmt2"):
            with self.assertRaisesRegex(ValueError, "COMPUTE_ALLOCATION"):
                self._run([], asset_trust_env=True)
        self.assertFalse((self.root / "asset").exists())

    def test_bulk_rejects_network_filesystem_and_insufficient_disk(self):
        with patch.object(assets.subprocess, "run", return_value=MagicMock(stdout="nfs4\n")):
            with self.assertRaisesRegex(ValueError, "NODE_LOCAL_STORAGE"):
                self._run([], asset_trust_env=True)
        with patch.object(assets.shutil, "disk_usage", return_value=MagicMock(free=10)):
            with self.assertRaisesRegex(ValueError, "DISK_ALLOWANCE"):
                self._run([], asset_trust_env=True, disk_reserve_bytes=3)
        self.assertFalse((self.root / "asset").exists())

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "asset"
        output.mkdir()
        (output / "edges.h5").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self._run([], asset_trust_env=True)
        self.assertEqual((output / "edges.h5").read_bytes(), b"preserve")

    def test_explicit_attempt_timeout_preserves_verified_prefix(self):
        with patch.object(assets.time, "monotonic", side_effect=[0, 1, 1, 1, 20]):
            with self.assertRaisesRegex(TimeoutError, "ASSET_TRANSFER_TIME_LIMIT"):
                self._run([self._response(b"abcd", 0, 3)], range_bytes=4, attempt_seconds=10)
        report = json.loads((self.root / "asset/source.json").read_text())
        self.assertEqual(report["verified_prefix_bytes"], 4)
        self.assertEqual((self.root / "asset/edges.h5.part").read_bytes(), b"abcd")

    def test_metadata_limits_cannot_be_raised(self):
        for options in ({"attempt_seconds": 241}, {"range_bytes": 512 * 1024 + 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "BOUND_CANNOT_BE_INCREASED"):
                self._run([], metadata_transfer=True, **options)
        self.assertFalse((self.root / "asset").exists())

    def test_proxy_exception_credentials_are_not_reported(self):
        error = requests.exceptions.ProxyError("connection to https://user:secret@proxy.invalid refused")
        with self.assertRaisesRegex(RuntimeError, "V1_ASSET_TRANSPORT_FAILED:ProxyError") as raised:
            self._run([error], asset_trust_env=True)
        report_text = (self.root / "asset/source.json").read_text()
        self.assertNotIn("secret", str(raised.exception))
        self.assertNotIn("secret", report_text)
        self.assertEqual(json.loads(report_text)["error_type"], "ProxyError")

    def test_batch_compute_context_does_not_require_an_srun_step_id(self):
        with patch.dict(os.environ, {"SLURM_STEP_ID": ""}):
            report, _ = self._run([self._response(b"abcdefgh", 0, 7, status=200)])
        self.assertTrue(report["asset_body_acquired"])

    def test_http_status_is_retained_without_credential_bearing_error_text(self):
        response = MagicMock(status_code=403)
        error = requests.exceptions.HTTPError("private transport details", response=response)
        with self.assertRaisesRegex(RuntimeError, "TRANSPORT_FAILED:HTTPError"):
            self._run([error], asset_trust_env=True)
        report = json.loads((self.root / "asset/source.json").read_text())
        self.assertEqual(report["http_status"], 403)
        self.assertNotIn("private transport", json.dumps(report))

    def test_cli_routes_bulk_flags_and_marks_limited_attempt_nonzero(self):
        argv = ["v1_source_assets", "--source-receipt", str(self.receipt), "--output", str(self.root / "asset"),
                "--asset-trust-env", "--range-bytes", "4", "--max-transfer-bytes", "4",
                "--attempt-seconds", "20", "--disk-reserve-bytes", "1024"]
        with patch.object(sys, "argv", argv), patch.object(assets, "download_resolved", return_value={"status": "PARTIAL"}) as run:
            with patch.object(sys, "stdout", io.StringIO()), self.assertRaises(SystemExit) as raised:
                assets.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertTrue(run.call_args.kwargs["asset_trust_env"])
        self.assertEqual(run.call_args.kwargs["range_bytes"], 4)
        self.assertEqual(run.call_args.kwargs["max_transfer_bytes"], 4)
        self.assertEqual(run.call_args.kwargs["attempt_seconds"], 20)
        self.assertEqual(run.call_args.kwargs["disk_reserve_bytes"], 1024)

    def test_cli_rejects_download_flags_without_source_receipt(self):
        argv = ["v1_source_assets", "--output", str(self.root / "asset"), "--probe-bytes", "4"]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                assets.main()
        self.assertEqual(raised.exception.code, 2)


    def _broken_response(self, tail=b"ef", error=None):
        response = self._response(tail, 4, 7)
        def broken(_):
            yield tail
            raise error if error is not None else requests.exceptions.ChunkedEncodingError("broken stream")
        response.iter_content.side_effect = broken
        return response

    def test_http502_retries_the_identical_range_and_retains_failure(self):
        failed = self._response(b"", 0, 3, status=502)
        failed.raise_for_status.side_effect = requests.HTTPError("bad gateway", response=failed)
        with patch.object(assets.time, "sleep") as sleep:
            result, session = self._run([failed, self._response(b"abcd", 0, 3),
                                        self._response(b"efgh", 4, 7)], range_bytes=4, range_retries=2)
        self.assertTrue(result["asset_body_acquired"])
        self.assertEqual(session.get.call_args_list[0], session.get.call_args_list[1])
        self.assertEqual(result["range_transfer_failures"], [dict(start=0, stop=3, failed_attempt=1,
            error_type="HTTPError", http_status=502, unverified_bytes=0, discarded_bytes=0, retry_scheduled=True)])
        sleep.assert_called_once_with(1.0)

    def test_two_partial_stream_retries_restore_both_hash_states(self):
        self.source["expected_md5"] = hashlib.md5(b"abcdefgh").hexdigest()
        self._save_source()
        with patch.object(assets.time, "sleep") as sleep:
            result, session = self._run([self._response(b"abcd", 0, 3), self._broken_response(b"e"),
                self._broken_response(b"ef"), self._response(b"efgh", 4, 7)], range_bytes=4, range_retries=2)
        self.assertEqual((self.root / "asset/edges.h5").read_bytes(), b"abcdefgh")
        self.assertEqual(result["sha256"], hashlib.sha256(b"abcdefgh").hexdigest())
        self.assertTrue(result["checksum_verification"]["md5"]["verified"])
        self.assertEqual([row["unverified_bytes"] for row in result["range_transfer_failures"]], [1, 2])
        self.assertEqual([row["discarded_bytes"] for row in result["range_transfer_failures"]], [1, 2])
        self.assertEqual([call.kwargs["headers"]["Range"] for call in session.get.call_args_list],
                         ["bytes=0-3", "bytes=4-7", "bytes=4-7", "bytes=4-7"])
        self.assertEqual([call.args for call in sleep.call_args_list], [(1.0,), (2.0,)])

    def test_exhausted_retries_preserve_terminal_tail_and_verified_prefix(self):
        session = MagicMock()
        session.get.side_effect = [self._response(b"abcd", 0, 3)] + [self._broken_response()] * 3
        with patch.object(assets.requests, "Session", return_value=session), patch.object(assets.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "V1_ASSET_TRANSPORT_FAILED"):
                download_resolved(self.receipt, self.root / "asset", range_bytes=4, range_retries=2)
        result = json.loads((self.root / "asset/source.json").read_text())
        self.assertEqual((result["received_bytes"], result["verified_prefix_bytes"]), (6, 4))
        self.assertEqual(result["verified_prefix_sha256"], hashlib.sha256(b"abcd").hexdigest())
        self.assertEqual((self.root / "asset/edges.h5.part").read_bytes(), b"abcdef")
        self.assertEqual(session.get.call_count, 4)
        self.assertEqual([row["retry_scheduled"] for row in result["range_transfer_failures"]], [True, True, False])

    def test_identity_and_tls_errors_do_not_retry(self):
        for label in ("etag", "tls"):
            with self.subTest(label=label):
                response = self._response(b"abcd", 0, 3, etag="changed")
                session = MagicMock()
                session.get.side_effect = ([response] if label == "etag" else requests.exceptions.SSLError("certificate"))
                with patch.object(assets.requests, "Session", return_value=session), patch.object(assets.time, "sleep") as sleep:
                    with self.assertRaises((ValueError, RuntimeError)):
                        download_resolved(self.receipt, self.root / label, range_bytes=4, range_retries=2)
                self.assertEqual(session.get.call_count, 1)
                sleep.assert_not_called()

    def test_deadline_prevents_backoff_and_another_request(self):
        failed = self._response(b"", 0, 3, status=502)
        failed.raise_for_status.side_effect = requests.HTTPError("bad gateway", response=failed)
        session = MagicMock(); session.get.return_value = failed
        with patch.object(assets.requests, "Session", return_value=session), \
             patch.object(assets.time, "monotonic", return_value=0), patch.object(assets.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "V1_ASSET_TRANSPORT_FAILED"):
                download_resolved(self.receipt, self.root / "asset", range_bytes=4, range_retries=2, attempt_seconds=0.5)
        self.assertEqual(session.get.call_count, 1)
        sleep.assert_not_called()

    def test_interrupt_bypasses_retries_and_still_saves_prefix(self):
        session = MagicMock()
        session.get.side_effect = [self._response(b"abcd", 0, 3), self._broken_response(error=KeyboardInterrupt())]
        with patch.object(assets.requests, "Session", return_value=session), patch.object(assets.time, "sleep") as sleep:
            with self.assertRaises(KeyboardInterrupt):
                download_resolved(self.receipt, self.root / "asset", range_bytes=4, range_retries=2)
        result = json.loads((self.root / "asset/source.json").read_text())
        self.assertEqual(result["error"], "V1_SOURCE_INTERRUPTED")
        self.assertEqual((result["received_bytes"], result["verified_prefix_bytes"]), (6, 4))
        self.assertEqual(session.get.call_count, 2)
        sleep.assert_not_called()

    def test_interrupt_during_backoff_leaves_only_the_verified_prefix(self):
        session = MagicMock()
        session.get.side_effect = [self._response(b"abcd", 0, 3), self._broken_response()]
        with patch.object(assets.requests, "Session", return_value=session), \
             patch.object(assets.time, "sleep", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                download_resolved(self.receipt, self.root / "asset", range_bytes=4, range_retries=2)
        result = json.loads((self.root / "asset/source.json").read_text())
        self.assertEqual((result["received_bytes"], result["verified_prefix_bytes"]), (4, 4))
        self.assertEqual((self.root / "asset/edges.h5.part").read_bytes(), b"abcd")
        self.assertEqual(result["range_transfer_failures"][0]["discarded_bytes"], 2)
        self.assertEqual(session.get.call_count, 2)

    def test_retry_budget_is_explicit_and_bounded(self):
        for count in (-1, 3, True, 0.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "RANGE_RETRIES"):
                download_resolved(self.receipt, self.root / "asset", range_bytes=4, range_retries=count)
        with self.assertRaisesRegex(ValueError, "RETRIES_REQUIRE_RANGED"):
            download_resolved(self.receipt, self.root / "asset", range_retries=2)


class ExplicitAddressAdapterTests(unittest.TestCase):
    def test_real_pool_keeps_url_host_tls_name_and_certificate_verification(self):
        url = "https://source.example.invalid:8443/records/42/files/source.zip?download=1"
        adapter = assets._ResolvedAddressAdapter(url, "192.0.2.10")
        self.addCleanup(adapter.close)
        prepared = requests.Request("GET", url).prepare()
        pool = adapter.get_connection_with_tls_context(prepared, verify=True)
        self.assertEqual((pool.host, pool.port), ("192.0.2.10", 8443))
        connection = pool._new_conn()
        self.assertEqual(connection.host, "192.0.2.10")
        self.assertEqual(connection.server_hostname, "source.example.invalid")
        self.assertEqual(connection.assert_hostname, "source.example.invalid")
        self.assertEqual(connection.cert_reqs, "CERT_REQUIRED")
        response = HTTPResponse(body=b"fixture", status=200)
        with patch.object(pool, "urlopen", return_value=response) as send:
            adapter.send(prepared, verify=True, proxies={})
        self.assertEqual(prepared.url, url)
        self.assertEqual(prepared.headers["Host"], "source.example.invalid:8443")
        self.assertEqual(send.call_args.kwargs["url"], "/records/42/files/source.zip?download=1")
        self.assertEqual(send.call_args.kwargs["headers"]["Host"], "source.example.invalid:8443")

    def test_invalid_source_origin_or_address_is_rejected(self):
        for url in ("http://source.example.invalid/file", "https:///file", "https://127.0.0.1/file",
                    "https://user:secret@source.example.invalid/file", "https://bad..host/file",
                    "https://source.example.invalid:invalid/file"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "HTTPS_SOURCE_HOST"):
                assets._ResolvedAddressAdapter(url, "192.0.2.10")
        for address in ("not-an-address", "https://192.0.2.10", "0.0.0.0", "::", "ff02::1", "fe80::1%eth0", 123):
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, "IP_ADDRESS"):
                assets._ResolvedAddressAdapter("https://source.example.invalid/file", address)

    def test_proxy_and_disabled_certificate_verification_are_rejected(self):
        url = "https://source.example.invalid/file"
        adapter = assets._ResolvedAddressAdapter(url, "192.0.2.10")
        self.addCleanup(adapter.close)
        prepared = requests.Request("GET", url).prepare()
        with self.assertRaisesRegex(ValueError, "CONFLICTS_WITH_PROXY_TRANSPORT"):
            adapter.get_connection_with_tls_context(prepared, verify=True, proxies={"https": "http://proxy.invalid"})
        for verify in (False, None, ""):
            with self.subTest(verify=verify), self.assertRaisesRegex(ValueError, "CERTIFICATE_VERIFICATION"):
                adapter.get_connection_with_tls_context(prepared, verify=verify)

    def test_redirect_to_other_origin_or_http_is_rejected(self):
        adapter = assets._ResolvedAddressAdapter("https://source.example.invalid/file", "192.0.2.10")
        self.addCleanup(adapter.close)
        for url in ("https://other.example.invalid/file", "http://source.example.invalid/file",
                    "https://source.example.invalid:8443/file"):
            prepared = requests.Request("GET", url).prepare()
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "SOURCE_ORIGIN_CHANGED"):
                adapter.get_connection_with_tls_context(prepared, verify=True)

    def test_ipv6_route_preserves_original_dns_hostname(self):
        adapter = assets._ResolvedAddressAdapter("https://source.example.invalid/file", "2001:db8::10")
        self.addCleanup(adapter.close)
        prepared = requests.Request("GET", "https://source.example.invalid/file").prepare()
        pool = adapter.get_connection_with_tls_context(prepared, verify=True)
        self.assertEqual(pool.host, "2001:db8::10")
        self.assertEqual(pool.conn_kw["server_hostname"], "source.example.invalid")
        self.assertEqual(pool.assert_hostname, "source.example.invalid")
        self.assertEqual(prepared.headers["Host"], "source.example.invalid")


if __name__ == "__main__":
    unittest.main()
