"""Tests for publication recovery; no source simulation is executed."""
import hashlib
import io
import json
import subprocess
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from population_dynamics.pfc_merlin_publication import get_optional, verify_bundle, recover_task


class PublicationRecoveryTests(unittest.TestCase):
    def fixture(self, root, *, exit_code=0, changed_manifest=False, duplicate=False,
                missing_terminal=False, duplicate_terminal=False, corrupted_terminal=False):
        task = dict(index=53, xie_job="652562_53", run_id="original",
                    trial_id="435964747", checkpoint_path="hdfs://source/task_0053")
        request = dict(task)
        request.pop("checkpoint_path")
        request.update(execution_backend="merlin", hostname="source-host",
            compatibility_execution_id="merlin_435964747_pfc_0053", status="RUNNING")
        terminal = dict(request, status="COMPLETE" if exit_code == 0 else "FAILED",
                        exit_code=exit_code)
        contents = {
            "tmp/runtime/tasks/53/request.json": json.dumps(request).encode(),
            "tmp/runtime/tasks/53/terminal.json": json.dumps(terminal).encode(),
            "share/results/source.bin": b"frozen source result",
        }
        if missing_terminal:
            contents.pop("tmp/runtime/tasks/53/terminal.json")
        if duplicate_terminal:
            contents["other/runtime/tasks/53/terminal.json"] = json.dumps(terminal).encode()
        manifest = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                    for name, data in contents.items()}
        if corrupted_terminal:
            contents["tmp/runtime/tasks/53/terminal.json"] = json.dumps(dict(terminal, corrupted=True)).encode()
        embedded = dict(manifest)
        if changed_manifest:
            embedded["invented"] = dict(bytes=0, sha256="bad")
        contents["merlin_return_manifest.json"] = json.dumps(embedded).encode()
        bundle = root / "result.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name, data in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if duplicate:
                name = "share/results/source.bin"
                member = tarfile.TarInfo(name)
                member.size = len(contents[name])
                archive.addfile(member, io.BytesIO(contents[name]))
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "terminal.json").write_text(json.dumps(terminal))
        return task, manifest, terminal, bundle

    def optional_files(self, *, receipt=None):
        def read(uri, destination):
            if uri.endswith("/receipt.json"):
                return receipt
            return json.loads(destination.read_text())
        return read

    def test_complete_source_preserves_identity_and_publishes_receipt_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, manifest, terminal, bundle = self.fixture(root)
            commands = []
            snapshots = {}
            def fake_hdfs(*args):
                commands.append(args)
                if args[0] == "-put" and str(args[1]).endswith(".json"):
                    snapshots[args[-1]] = json.loads(Path(args[1]).read_text())
            publisher = dict(run_id="publisher", trial_id="new-trial",
                             hostname="publisher-host", checkpoint_path="hdfs://new")
            with patch("population_dynamics.pfc_merlin_publication.hdfs", side_effect=fake_hdfs), \
                    patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=self.optional_files()):
                result = recover_task(task, root, publisher)
            self.assertEqual(result["status"], "PUBLICATION_RECOVERED")
            self.assertEqual(commands[-1][-1], result["receipt_uri"])
            receipt = snapshots[result["receipt_uri"]]
            self.assertEqual(receipt["run_id"], "original")
            self.assertEqual(receipt["hostname"], "source-host")
            self.assertEqual(receipt["publication_recovery"]["run_id"], "publisher")
            self.assertEqual(receipt["publication_recovery"]["source_simulations"], 0)
            self.assertTrue(all("-f" not in command for command in commands))
            self.assertTrue(any(command[0] == "-put" and command[-1] ==
                receipt["checkpoint_path"] + "/result.tar.gz" for command in commands))
            self.assertEqual(verify_bundle(bundle, manifest, terminal, task)["verified_files"], 3)

    def test_invalid_source_never_publishes(self):
        for kind in ("source_failure", "manifest_changed", "duplicate", "identity_changed"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task, manifest, terminal, bundle = self.fixture(root,
                    exit_code=1 if kind == "source_failure" else 0,
                    changed_manifest=kind == "manifest_changed", duplicate=kind == "duplicate")
                if kind == "identity_changed":
                    task["trial_id"] = "wrong"
                with patch("population_dynamics.pfc_merlin_publication.hdfs") as run_hdfs, \
                        patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=self.optional_files()), \
                        self.assertRaises(ValueError):
                    recover_task(task, root, dict(checkpoint_path="hdfs://new"))
                self.assertFalse(any(args.args[0] == "-put" for args in run_hdfs.call_args_list))

    def test_existing_verified_publication_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, manifest, terminal, bundle = self.fixture(root)
            verified = verify_bundle(bundle, manifest, terminal, task)
            receipt = dict(terminal, persisted=True, checkpoint_path=task["checkpoint_path"],
                result_sha256=verified["sha256"], result_bytes=verified["bytes"],
                chunks=[dict(name="chunk_00000.json", bytes=verified["bytes"],
                             sha256=verified["sha256"])])
            original_reader = self.optional_files(receipt=receipt)
            def optional(uri, destination):
                if uri.endswith("/chunk_00000.json"):
                    import base64
                    return dict(content_base64=base64.b64encode(bundle.read_bytes()).decode())
                return original_reader(uri, destination)
            with patch("population_dynamics.pfc_merlin_publication.hdfs") as run_hdfs, \
                    patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=optional):
                result = recover_task(task, root, dict(checkpoint_path="hdfs://new"))
            self.assertEqual(result["status"], "EXISTING_PUBLICATION_REUSED")
            self.assertFalse(any(args.args[0] == "-put" for args in run_hdfs.call_args_list))

    def test_missing_or_corrupt_receipt_chunk_republishes_verified_bundle(self):
        for failure in ("missing", "corrupt", "invalid_json", "whole_digest"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task, manifest, terminal, bundle = self.fixture(root)
                verified = verify_bundle(bundle, manifest, terminal, task)
                receipt = dict(terminal, persisted=True, checkpoint_path=task["checkpoint_path"],
                    result_sha256=verified["sha256"], result_bytes=verified["bytes"],
                    chunks=[dict(name="chunk_00000.json", bytes=verified["bytes"],
                                 sha256=verified["sha256"])])
                alternate = bundle.read_bytes()[:-1] + bytes([bundle.read_bytes()[-1] ^ 1])
                if failure == "whole_digest":
                    receipt["chunks"][0]["sha256"] = hashlib.sha256(alternate).hexdigest()
                original_reader = self.optional_files(receipt=receipt)
                published = {}
                def optional(uri, destination):
                    if uri.endswith("/chunk_00000.json"):
                        if failure == "invalid_json":
                            raise json.JSONDecodeError("truncated", "", 0)
                        if failure == "whole_digest":
                            import base64
                            return dict(content_base64=base64.b64encode(alternate).decode())
                        return None if failure == "missing" else dict(content_base64="YmFk")
                    return original_reader(uri, destination)
                def fake_hdfs(*args):
                    if args[0] == "-put" and str(args[1]).endswith(".json"):
                        published[args[-1]] = json.loads(Path(args[1]).read_text())
                with patch("population_dynamics.pfc_merlin_publication.hdfs", side_effect=fake_hdfs), \
                        patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=optional):
                    result = recover_task(task, root, dict(run_id="publisher", checkpoint_path="hdfs://new"))
                self.assertEqual(result["status"], "PUBLICATION_RECOVERED")
                restored = published[result["receipt_uri"]]
                self.assertTrue(restored["publication_recovery"]["original_receipt_present"])
                validation = restored["publication_recovery"]["source_transport_validation"]
                self.assertEqual(validation["status"], "FAILED")
                self.assertEqual(validation["failures"][0]["reason"],
                    {"missing": "MISSING_CHUNK", "corrupt": "CHUNK_BYTES_OR_SHA256_MISMATCH",
                     "invalid_json": "INVALID_CHUNK_JSON", "whole_digest": "ASSEMBLED_BUNDLE_SHA256_MISMATCH"}[failure])
                self.assertEqual(restored["result_sha256"], verified["sha256"])
                self.assertEqual(restored["run_id"], task["run_id"])

    def test_uncertain_receipt_chunk_read_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, manifest, terminal, bundle = self.fixture(root)
            verified = verify_bundle(bundle, manifest, terminal, task)
            receipt = dict(terminal, persisted=True, checkpoint_path=task["checkpoint_path"],
                result_sha256=verified["sha256"], result_bytes=verified["bytes"],
                chunks=[dict(name="chunk_00000.json", bytes=verified["bytes"], sha256=verified["sha256"])])
            original_reader = self.optional_files(receipt=receipt)
            def optional(uri, destination):
                if uri.endswith("/chunk_00000.json"):
                    raise RuntimeError("HDFS_RECEIPT_READ_FAILED:Connection timed out")
                return original_reader(uri, destination)
            with patch("population_dynamics.pfc_merlin_publication.hdfs") as run_hdfs, \
                    patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=optional), \
                    self.assertRaisesRegex(RuntimeError, "Connection timed out"):
                recover_task(task, root, dict(checkpoint_path="hdfs://new"))
            self.assertFalse(any(call.args[0] == "-put" for call in run_hdfs.call_args_list))


    def test_complete_embedded_metadata_preserves_source_identity(self):
        for missing in (("manifest", "terminal"), ("manifest",), ("terminal",)):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task, manifest, terminal, bundle = self.fixture(root)
                request = dict(terminal, status="RUNNING")
                request.pop("exit_code")
                for role in missing:
                    (root / (role + ".json")).unlink()
                commands, published = [], {}

                def fake_hdfs(*args):
                    commands.append(args)
                    if args[0] == "-get" and str(args[1]).endswith("/request.json"):
                        Path(args[2]).write_text(json.dumps(request))
                    if args[0] == "-put" and str(args[1]).endswith(".json"):
                        published[args[-1]] = json.loads(Path(args[1]).read_text())

                def optional(uri, destination):
                    if uri.endswith("/receipt.json") or Path(uri).stem in missing:
                        return None
                    return json.loads(destination.read_text())

                publisher = dict(run_id="publisher", trial_id="new-trial",
                                 hostname="publisher-host", checkpoint_path="hdfs://new")
                with patch("population_dynamics.pfc_merlin_publication.hdfs", side_effect=fake_hdfs), \
                        patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=optional):
                    result = recover_task(task, root, publisher)
                receipt = published[result["receipt_uri"]]
                self.assertEqual(receipt["run_id"], task["run_id"])
                self.assertEqual(receipt["trial_id"], task["trial_id"])
                self.assertEqual(receipt["hostname"], "source-host")
                self.assertEqual(receipt["publication_recovery"]["source_metadata_origin"], {
                    "manifest": "embedded_bundle" if "manifest" in missing else "external_hdfs",
                    "terminal": "embedded_bundle" if "terminal" in missing else "external_hdfs",
                    "request": "external_hdfs",
                })
                self.assertEqual(commands[0][1], task["checkpoint_path"] + "/result.tar.gz")
                self.assertEqual(commands[-1][-1], result["receipt_uri"])
                self.assertEqual(verify_bundle(bundle, manifest, terminal, task)["verified_files"], 3)

    def test_invalid_embedded_terminal_or_request_never_publishes(self):
        for kind, error in (("missing_terminal", "SOURCE_TERMINAL_MISSING"),
                            ("duplicate_terminal", "DUPLICATE_TASK_TERMINAL"),
                            ("corrupted_terminal", "ARCHIVE_FILE_CHANGED"),
                            ("request_changed", "SOURCE_EXTERNAL_REQUEST_BINDING_MISMATCH")):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task, _, terminal, _ = self.fixture(root,
                    missing_terminal=kind == "missing_terminal",
                    duplicate_terminal=kind == "duplicate_terminal",
                    corrupted_terminal=kind == "corrupted_terminal")
                (root / "manifest.json").unlink()
                (root / "terminal.json").unlink()
                request = dict(terminal, status="RUNNING")
                request.pop("exit_code")
                if kind == "request_changed":
                    request["hostname"] = "wrong-host"

                def fake_hdfs(*args):
                    if args[0] == "-get" and str(args[1]).endswith("/request.json"):
                        Path(args[2]).write_text(json.dumps(request))

                with patch("population_dynamics.pfc_merlin_publication.hdfs", side_effect=fake_hdfs) as run_hdfs, \
                        patch("population_dynamics.pfc_merlin_publication.get_optional", return_value=None), \
                        self.assertRaisesRegex(ValueError, error):
                    recover_task(task, root, dict(checkpoint_path="hdfs://new"))
                self.assertFalse(any(call.args[0] == "-put" for call in run_hdfs.call_args_list))

    def test_truncated_or_missing_bundle_never_publishes(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task, _, _, bundle = self.fixture(root)
                if not missing:
                    bundle.write_bytes(b"truncated gzip bundle")

                def fake_hdfs(*args):
                    if missing and args[0] == "-get":
                        raise subprocess.CalledProcessError(1, ["hdfs"], stderr="File does not exist")

                with patch("population_dynamics.pfc_merlin_publication.hdfs", side_effect=fake_hdfs) as run_hdfs, \
                        patch("population_dynamics.pfc_merlin_publication.get_optional", return_value=None), \
                        self.assertRaises((tarfile.TarError, EOFError, OSError, subprocess.CalledProcessError)):
                    recover_task(task, root, dict(checkpoint_path="hdfs://new"))
                self.assertFalse(any(call.args[0] == "-put" for call in run_hdfs.call_args_list))

    def test_explicit_upload_object_requires_complete_gzip(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task, _, _, bundle = self.fixture(root)
                task["source_bundle_uri"] = task["checkpoint_path"] + "/result.tar.gz._COPYING_"
                if truncated:
                    bundle.write_bytes(bundle.read_bytes()[:-8])
                commands = []
                def fake_hdfs(*args):
                    commands.append(args)
                with patch("population_dynamics.pfc_merlin_publication.hdfs", side_effect=fake_hdfs), \
                        patch("population_dynamics.pfc_merlin_publication.get_optional", side_effect=self.optional_files()):
                    if truncated:
                        with self.assertRaises((EOFError, OSError, tarfile.TarError)):
                            recover_task(task, root, dict(checkpoint_path="hdfs://new"))
                        self.assertFalse(any(command[0] == "-put" for command in commands))
                    else:
                        result = recover_task(task, root, dict(checkpoint_path="hdfs://new"))
                        self.assertEqual(result["status"], "PUBLICATION_RECOVERED")
                        self.assertEqual(commands[0][1], task["source_bundle_uri"])
                        self.assertTrue(all("-f" not in command for command in commands))

    def test_optional_metadata_falls_back_only_for_explicit_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "manifest.json"
            for message in ("No such file or directory", "File does not exist", "FileNotFoundException"):
                with self.subTest(message=message), patch(
                        "population_dynamics.pfc_merlin_publication.subprocess.run",
                        return_value=subprocess.CompletedProcess(["hdfs"], 1, stdout="", stderr=message)):
                    self.assertIsNone(get_optional("hdfs://source/manifest.json", destination))
            for message in ("Permission denied", "Connection timed out", "unknown failure"):
                with self.subTest(message=message), patch(
                        "population_dynamics.pfc_merlin_publication.subprocess.run",
                        return_value=subprocess.CompletedProcess(["hdfs"], 1, stdout="", stderr=message)), \
                        self.assertRaisesRegex(RuntimeError, "HDFS_RECEIPT_READ_FAILED"):
                    get_optional("hdfs://source/manifest.json", destination)


if __name__ == "__main__":
    unittest.main()
