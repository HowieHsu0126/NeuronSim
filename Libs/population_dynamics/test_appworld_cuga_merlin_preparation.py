"""Focused CUGA preparation checks; run only in a Merlin CPU allocation.

No real model, task, package installation, HDFS call, or legacy launcher executes.
The small archive fixtures check the actual persistence formats.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts import prepare_appworld_cuga_merlin as preparation


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_allocation_requires_real_merlin_cpu_identity(self):
        valid = {
            "RH2_JOB_RUN_ID": "fixture-run", "ARNOLD_TRIAL_ID": "123",
            "ARNOLD_APPLIED_GPU_NUM": "0", "ARNOLD_HEAD_CPU": "4",
            "RH2_CHECKPOINT_HDFS": "hdfs://fixture/checkpoints/fixture-run",
        }
        with mock.patch.object(preparation.socket, "gethostname", return_value="fixture-worker"):
            identity = preparation.require_merlin_cpu(valid)
            self.assertEqual(identity["allocated_gpus"], 0)
            self.assertEqual(identity["execution_backend"], "merlin")
            self.assertEqual(identity["compatibility_execution_id"], "merlin_123_cuga")
            for change in (
                {"ARNOLD_APPLIED_GPU_NUM": "1"}, {"ARNOLD_APPLIED_GPU_NUM": ""},
                {"RH2_JOB_RUN_ID": ""}, {"ARNOLD_TRIAL_ID": ""},
                {"SLURM_JOB_ID": "123"}, {"RH2_CHECKPOINT_HDFS": "hdfs://fixture/wrong"},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    preparation.require_merlin_cpu({**valid, **change})
        with mock.patch.object(preparation.socket, "gethostname", return_value="mgmt2"):
            with self.assertRaises(ValueError):
                preparation.require_merlin_cpu(valid)

    def test_bootstrap_selects_private_python_and_pip_without_network_or_system_install(self):
        output = self.root / "artifacts"
        output.mkdir()
        uploads = []

        class Store:
            def publish(self, path, relative):
                uploads.append(relative)
                return {"uri": "hdfs://fixture/" + relative, **preparation.file_identity(path)}

        report = {}
        original_prefix = sys.prefix
        environment = preparation.prepare_bootstrap_environment(
            dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                 PYTHONDONTWRITEBYTECODE="1", UV_CACHE_DIR="/nonwritable-fixture/uv-cache"),
            work=self.root, output=output, store=Store(), report=report,
        )
        result = subprocess.run(
            ["python3", "-c",
             "import json,pip,sys; print(json.dumps(dict(executable=sys.executable,"
             "prefix=sys.prefix,pip_file=pip.__file__)))"],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads(result.stdout)
        private = self.root / "bootstrap-env"
        self.assertEqual(Path(identity["prefix"]), private)
        self.assertEqual(Path(identity["executable"]).parent, private / "bin")
        self.assertTrue(Path(identity["pip_file"]).is_relative_to(private))
        self.assertNotEqual(identity["prefix"], original_prefix)
        self.assertEqual(sys.prefix, original_prefix)
        self.assertEqual(environment["VIRTUAL_ENV"], str(private))
        self.assertEqual(environment["UV_CACHE_DIR"], str(self.root / "uv-cache"))
        self.assertTrue((self.root / "uv-cache").is_dir())
        stage = report["stages"]["create_bootstrap_environment"]
        self.assertEqual(stage["command"], [sys.executable, "-m", "venv", str(private)])
        self.assertEqual(stage["exit_code"], 0)
        self.assertTrue(stage["cleanup"]["owner_terminated"])
        self.assertTrue(stage["outputs_closed"])
        self.assertIn("stages/stage_receipts/create_bootstrap_environment.json", uploads)

    def test_prefix_preserves_recipe_and_exit_code_without_old_tests_or_publication(self):
        prefix = '#!/bin/bash\nset -euo pipefail\nset +e\nfalse\ncode=$?'
        original = (prefix + '\nif [ "$code" = 0 ]; then echo old_tests; fi\necho old_publication\n').encode()
        path = self.root / "prior.sh"
        path.write_bytes(original)
        with (
            mock.patch.object(preparation, "PRIOR_LAUNCHER_BYTES", len(original)),
            mock.patch.object(preparation, "PRIOR_LAUNCHER_SHA256", hashlib.sha256(original).hexdigest()),
        ):
            result = preparation.preparation_prefix(path)
            self.assertEqual(result, (prefix + '\nexit "$code"\n').encode())
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(original.replace(b"false", b"true "))
            with self.assertRaisesRegex(ValueError, "IDENTITY_DIFFERS"):
                preparation.preparation_prefix(path)


        malformed = [
            ((prefix + "\necho no_boundary\n").encode(), "BOUNDARY_DIFFERS"),
            ((prefix + preparation.PREPARATION_BOUNDARY + "true; fi"
              + preparation.PREPARATION_BOUNDARY + "true; fi\n").encode(), "BOUNDARY_DIFFERS"),
            ((prefix.replace("code=$?", "code=0") + preparation.PREPARATION_BOUNDARY
              + "true; fi\n").encode(), "EXIT_CAPTURE_DIFFERS"),
        ]
        for payload, reason in malformed:
            path.write_bytes(payload)
            with (
                self.subTest(reason=reason),
                mock.patch.object(preparation, "PRIOR_LAUNCHER_BYTES", len(payload)),
                mock.patch.object(preparation, "PRIOR_LAUNCHER_SHA256", hashlib.sha256(payload).hexdigest()),
                self.assertRaisesRegex(ValueError, reason),
            ):
                preparation.preparation_prefix(path)

    def test_current_source_manifest_rejects_missing_or_corrupt_input(self):
        def record(payload):
            return {"content_base64": base64.b64encode(payload).decode(),
                    "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        files = {name: record(b"fixture\n") for name in preparation.REQUIRED_FILES}
        with self.assertRaisesRegex(ValueError, "SOURCE_SET_REQUIRED"):
            preparation.stage_current_sources({"files": {}}, self.root / "missing")
        name = sorted(files)[0]
        files[name] = {**files[name], "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "DECLARED_EMBEDDED_BYTES_DIFFER"):
            preparation.stage_current_sources({"files": files}, self.root / "corrupt")

    def test_current_check_adaptation_preserves_same_host_and_resource_requirements(self):
        library = Path(__file__).resolve().parents[1]
        repository = self.root / "repository"
        relative_paths = [
            *(Path("Libs/scripts") / name for name in preparation.CHECK_SCRIPTS),
            Path("Libs/population_dynamics/resource_probe.py"),
        ]
        for relative in relative_paths:
            source = library.parent / relative
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        records = preparation.adapt_current_checks(repository)
        self.assertEqual(set(records), {path.as_posix() for path in relative_paths})
        for name in preparation.CHECK_SCRIPTS:
            path = repository / "Libs/scripts" / name
            text = path.read_text()
            self.assertIn("ARNOLD_APPLIED_GPU_NUM", text)
            self.assertIn("512*1024**2", text)
            self.assertIn("CUGA_PACKAGES_PREPARED_NATIVE_INTEGRATION_REQUIRED", text)
            self.assertIn(preparation.CUGA_REVISION, text)
            syntax = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
        graph = (repository / "Libs/scripts/validate_appworld_cuga_runtime.sbatch").read_text()
        self.assertIn("receipt.get('hostname', '').split('.')[0] != hostname", graph)
        probe = (repository / "Libs/population_dynamics/resource_probe.py").read_text()
        compile(probe, "adapted_resource_probe", "exec")
        self.assertIn('os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0"', probe)
        with self.assertRaisesRegex(ValueError, "ANCHOR_DIFFERS"):
            preparation.adapt_current_checks(repository)

    def test_runtime_archive_preserves_prefixes_and_interpreter_symlink(self):
        source, base = self.root / "source", self.root / "managed-python"
        interpreter = base / "bin/python3.13"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(b"synthetic interpreter identity")
        link = source / ".venv/bin/python"
        link.parent.mkdir(parents=True)
        link.symlink_to(interpreter)
        (source / "uv.lock").write_text("synthetic lock")
        destination = self.root / "runtime.tar.gz"
        result = preparation.archive_runtime((source, base), destination)
        self.assertEqual(result["symlink_targets"], {str(link): str(interpreter)})
        with tarfile.open(destination, "r:gz") as archive:
            member = archive.getmember(link.as_posix().lstrip("/"))
            self.assertTrue(member.issym())
            self.assertEqual(member.linkname, str(interpreter))
            self.assertTrue(archive.getmember(interpreter.as_posix().lstrip("/")).isfile())

    def test_runtime_link_escape_rejects_before_creating_an_archive(self):
        source, base = self.root / "source", self.root / "base"
        source.mkdir()
        base.mkdir()
        outside = self.root / "outside"
        outside.write_text("not part of the runtime")
        (source / "escape").symlink_to(outside)
        destination = self.root / "runtime.tar.gz"
        with self.assertRaisesRegex(ValueError, "LINK_ESCAPES"):
            preparation.archive_runtime((source, base), destination)
        self.assertFalse(destination.exists())

    def test_evidence_format_matches_return_contract_and_keeps_failure_status(self):
        output = self.root / "artifacts"
        output.mkdir()
        (output / "environment.json").write_text('{"status":"fixture"}\n')
        (self.root / "runtime.tar.gz").write_bytes(b"large runtime remains separate")
        uploaded = {}

        class Store:
            def publish(self, path, relative):
                uploaded[relative] = path.read_bytes()
                return {"uri": "hdfs://fixture/" + relative, **preparation.file_identity(path)}

        report = {
            "run_id": "fixture-run", "trial_id": "fixture-trial", "status": "FAILED",
            "exit_code": 1, "persisted": False, "native_integration_qualified": False,
            "runtime_bundle": {"uri": "hdfs://fixture/runtime.tar.gz", "bytes": 123, "sha256": "a" * 64},
        }
        preparation.publish_evidence(Store(), self.root, output, report)
        receipt = json.loads(uploaded["receipt.json"])
        manifest = json.loads(uploaded["output_manifest.json"])
        self.assertEqual(receipt["status"], "FAILED")
        self.assertEqual(receipt["exit_code"], 1)
        self.assertFalse(receipt["native_integration_qualified"])
        self.assertTrue(receipt["persisted"])
        self.assertFalse(manifest["runtime_archives_included"])
        self.assertNotIn("runtime.tar.gz", manifest["files"])
        self.assertEqual(set(receipt["output_bundle"]), {"uri", "bytes", "sha256"})
        self.assertEqual(set(receipt["artifact_manifest"]), {"uri", "bytes", "sha256"})
        with tarfile.open(self.root / "result.tar.gz", "r:gz") as archive:
            members = archive.getmembers()
            self.assertTrue(all(member.isfile() for member in members))
            self.assertEqual({member.name for member in members},
                             {"output_manifest.json", *("artifacts/" + name for name in manifest["files"])})


    def runtime_fixture(self, base):
        source = self.root / "scratch/source"
        source.mkdir(parents=True, exist_ok=True)
        base.mkdir(parents=True, exist_ok=True)
        python = str(source / ".venv/bin/python")
        environment = {
            "status": preparation.PREPARATION_STATUS,
            "source_revision": preparation.CUGA_REVISION,
            "source_checkout": str(source), "python": python,
            "python_version": "3.13.12", "packages": {"cuga": "fixture-version"},
            "hostname": "origin-worker", "job_id": "merlin_origin_cuga",
        }
        identity = {
            "executable": python, "prefix": str(source / ".venv"),
            "base_prefix": str(base), "source_checkout": str(source),
            "python_version": environment["python_version"],
            "packages": environment["packages"], "hostname": "origin-worker",
        }
        return source, environment, identity

    def test_image_base_requires_exact_python_source_and_package_identity(self):
        base = self.root / "image/cpython-3.13.12-linux-x86_64-gnu"
        source, environment, identity = self.runtime_fixture(base)
        with (
            mock.patch.object(preparation, "LEGACY_SCRATCH", self.root / "scratch"),
            mock.patch.object(preparation, "IMAGE_PYTHON_BASE", base),
        ):
            self.assertEqual(preparation.runtime_roots(environment, identity), (source, base))
            with self.assertRaisesRegex(ValueError, "ROOT_DIFFERS"):
                preparation.runtime_roots(
                    {**environment, "python_version": "3.13.11"},
                    {**identity, "python_version": "3.13.11"},
                )
            with self.assertRaisesRegex(ValueError, "PINNED_CUGA_PREPARATION"):
                preparation.runtime_roots({**environment, "source_revision": "other"}, identity)
            for changed in (
                {"packages": {"cuga": "different"}},
                {"executable": str(source / ".venv/bin/other-python")},
                {"source_checkout": str(source.parent / "other-source")},
            ):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "PREFIX_IDENTITY"):
                    preparation.runtime_roots(environment, {**identity, **changed})

    def test_runtime_roots_reject_arbitrary_image_base(self):
        base = self.root / "image/unverified-python"
        _, environment, identity = self.runtime_fixture(base)
        with mock.patch.object(preparation, "LEGACY_SCRATCH", self.root / "scratch"):
            with self.assertRaisesRegex(ValueError, "ROOT_DIFFERS"):
                preparation.runtime_roots(environment, identity)

    def test_original_managed_python_base_remains_supported(self):
        base = self.root / "work/python/cpython-3.13.12-linux-x86_64-gnu"
        source, environment, identity = self.runtime_fixture(base)
        with (
            mock.patch.object(preparation, "LEGACY_SCRATCH", self.root / "scratch"),
            mock.patch.object(preparation, "LEGACY_WORK", self.root / "work"),
        ):
            self.assertEqual(preparation.runtime_roots(environment, identity), (source, base))

    def test_runtime_roots_reject_source_directory_alias(self):
        base = self.root / "image/cpython-3.13.12-linux-x86_64-gnu"
        source, environment, identity = self.runtime_fixture(base)
        alias = source.parent / "alias"
        alias.symlink_to(source, target_is_directory=True)
        environment = {**environment, "source_checkout": str(alias), "python": str(alias / ".venv/bin/python")}
        identity = {
            **identity, "source_checkout": str(alias), "prefix": str(alias / ".venv"),
            "executable": environment["python"],
        }
        with (
            mock.patch.object(preparation, "LEGACY_SCRATCH", self.root / "scratch"),
            mock.patch.object(preparation, "IMAGE_PYTHON_BASE", base),
        ):
            with self.assertRaisesRegex(ValueError, "ORIGINAL_REAL_DIRECTORY"):
                preparation.runtime_roots(environment, identity)


    def test_staging_prefix_excludes_installation_and_preserves_bootstrap(self):
        block = "python3 - <<'CUGA_BOOTSTRAP'\nprint('staged fixture')\nCUGA_BOOTSTRAP\n"
        payload = ("#!/bin/bash\npip install forbidden\n" + block
                   + "sudo -n mkdir -p /data/neuronsim\nsudo -n chown fixture:fixture /data/neuronsim\n"
                   + "set +e\nbash install_packages.sh\ncode=$?\n"
                   + 'if [ "$code" = 0 ]; then echo old_checks; fi\n').encode()
        path = self.root / "prior.sh"
        path.write_bytes(payload)
        with (
            mock.patch.object(preparation, "PRIOR_LAUNCHER_BYTES", len(payload)),
            mock.patch.object(preparation, "PRIOR_LAUNCHER_SHA256", hashlib.sha256(payload).hexdigest()),
        ):
            staged = preparation.staging_prefix(path).decode()
        self.assertIn(block, staged)
        self.assertIn("sudo -n mkdir -p /data/neuronsim", staged)
        self.assertIn("sudo -n chown fixture:fixture /data/neuronsim", staged)
        self.assertNotIn("pip install", staged)
        self.assertNotIn("install_packages.sh", staged)
        self.assertNotIn("old_checks", staged)

    def restore_fixture(self, image_alias=False):
        source, base = self.root / "runtime-source", self.root / "image-python"
        interpreter = base / "bin/python3.13"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(b"fixed image interpreter")
        interpreter.chmod(0o755)
        link = source / ".venv/bin/python"
        link.parent.mkdir(parents=True)
        alias_root = self.root / "image-python-alias"
        if image_alias:
            alias_root.symlink_to(base, target_is_directory=True)
        link.symlink_to(alias_root / "bin/python3.13" if image_alias else interpreter)
        (source / "uv.lock").write_text("fixed lock")
        archive = self.root / "saved.tar.gz"
        receipt = preparation.archive_runtime((source, base), archive)
        shutil.rmtree(source)
        work = self.root / "restore-work"
        work.mkdir()

        class Store:
            def hdfs(self, operation, uri, destination):
                if operation != "-get":
                    raise AssertionError(operation)
                shutil.copyfile(archive, destination)

        return source, base, interpreter, work, receipt, Store()

    def test_restore_reuses_identical_readonly_image_base_without_writing_it(self):
        source, base, interpreter, work, receipt, store = self.restore_fixture()
        interpreter.chmod(0o555)
        before = (interpreter.stat().st_ino, interpreter.stat().st_mtime_ns, interpreter.stat().st_mode)
        preparation.restore_archive(
            store, work, "restored", "hdfs://fixture/runtime", receipt,
            (source, base), (source, base), existing_identical_roots=(base,),
        )
        self.assertEqual((source / "uv.lock").read_text(), "fixed lock")
        self.assertTrue((source / ".venv/bin/python").is_symlink())
        self.assertEqual((source / ".venv/bin/python").resolve(), interpreter)
        self.assertEqual((interpreter.stat().st_ino, interpreter.stat().st_mtime_ns, interpreter.stat().st_mode), before)
        self.assertEqual(interpreter.read_bytes(), b"fixed image interpreter")

    def test_differing_image_base_rejects_before_source_extraction(self):
        source, base, interpreter, work, receipt, store = self.restore_fixture()
        interpreter.write_bytes(b"different image interpreter")
        with self.assertRaisesRegex(ValueError, "EXISTING_RUNTIME_ENTRY_DIFFERS"):
            preparation.restore_archive(
                store, work, "restored", "hdfs://fixture/runtime", receipt,
                (source, base), (source, base), existing_identical_roots=(base,),
            )
        self.assertFalse(source.exists())
        self.assertEqual(interpreter.read_bytes(), b"different image interpreter")

    def test_missing_image_base_member_rejects_before_source_extraction(self):
        source, base, interpreter, work, receipt, store = self.restore_fixture()
        interpreter.unlink()
        with self.assertRaisesRegex(ValueError, "EXISTING_RUNTIME_ENTRY_DIFFERS"):
            preparation.restore_archive(
                store, work, "restored", "hdfs://fixture/runtime", receipt,
                (source, base), (source, base), existing_identical_roots=(base,),
            )
        self.assertFalse(source.exists())
        self.assertFalse(interpreter.exists())

    def test_legacy_restore_still_rejects_an_existing_destination_root(self):
        source, base, _, work, receipt, store = self.restore_fixture()
        with self.assertRaisesRegex(ValueError, "RESTORE_PREFIX_ALREADY_EXISTS"):
            preparation.restore_archive(
                store, work, "restored", "hdfs://fixture/runtime", receipt,
                (source, base), (source, base),
            )
        self.assertFalse(source.exists())


    def test_existing_python_alias_is_preserved_only_with_exact_mapping(self):
        source, base, interpreter, work, receipt, store = self.restore_fixture(image_alias=True)
        alias = self.root / "image-python-alias" / "bin/python3.13"
        preparation.restore_archive(
            store, work, "restored", "hdfs://fixture/runtime", receipt,
            (source, base), (source, base), existing_identical_roots=(base,),
            existing_link_aliases={alias: interpreter},
        )
        self.assertEqual(os.readlink(source / ".venv/bin/python"), str(alias))
        self.assertEqual((source / ".venv/bin/python").resolve(), interpreter)
        self.assertTrue((self.root / "image-python-alias").is_symlink())

    def test_existing_python_alias_without_mapping_remains_rejected(self):
        source, base, _, work, receipt, store = self.restore_fixture(image_alias=True)
        with self.assertRaisesRegex(ValueError, "UNSAFE_ARCHIVE_LINK"):
            preparation.restore_archive(
                store, work, "restored", "hdfs://fixture/runtime", receipt,
                (source, base), (source, base), existing_identical_roots=(base,),
            )
        self.assertFalse(source.exists())

    def test_changed_python_alias_rejects_before_source_extraction(self):
        source, base, interpreter, work, receipt, store = self.restore_fixture(image_alias=True)
        alias_root = self.root / "image-python-alias"
        alias_root.unlink()
        other = self.root / "different-image"
        (other / "bin").mkdir(parents=True)
        (other / "bin/python3.13").write_bytes(b"unverified interpreter")
        alias_root.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "EXISTING_RUNTIME_LINK_ALIAS_DIFFERS"):
            preparation.restore_archive(
                store, work, "restored", "hdfs://fixture/runtime", receipt,
                (source, base), (source, base), existing_identical_roots=(base,),
                existing_link_aliases={alias_root / "bin/python3.13": interpreter},
            )
        self.assertFalse(source.exists())
        self.assertEqual(alias_root.resolve(), other)

    def test_restore_requires_a_persisted_package_runtime_before_fetching(self):
        for origin in ({}, {"persisted": True}, {
            "schema": "appworld_cuga_merlin_preparation_v1", "persisted": True,
            "runtime_prepared": True, "runtime_persisted": False,
        }):
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "PERSISTED_CUGA_RUNTIME"):
                preparation.restore_prepared_runtime(
                    origin, {}, work=self.root, output=self.root, store=None,
                    report={}, environment_path=self.root / "environment.json",
                )

    def test_runtime_identity_fragment_compiles_without_loading_runtime(self):
        compile(preparation.RUNTIME_IDENTITY, "<runtime_identity>", "exec")


if __name__ == "__main__":
    unittest.main()
