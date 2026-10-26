"""Exercise archived native-test reuse without downloads or Slurm execution."""

import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class V1SourceLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        launcher = (Path(__file__).resolve().parents[2]
                    / "Output/neural_population_dynamics/operations/v1_source_acquisition.sbatch")
        source = launcher.read_text(encoding="utf-8")
        boundary = 'interpreter=${NRNSIM_V1_PYTHON:?'
        if source.count(boundary) != 1:
            raise AssertionError("Expected placement checks before runtime setup")
        cls.placement_source = source.split(boundary, 1)[0] + "\nprintf '%s\\n' PLACEMENT_ACCEPTED\n"
        branch = source.split('if [[ -n "${NRNSIM_V1_TEST_ARCHIVE:-}" ]]; then', 1)[1]
        embedded = branch.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        cls.reuse_code = compile(embedded, str(launcher), "exec")
        signal_source = source.split("cat > \"$work/asset_cli.py\" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        tree = ast.parse(signal_source)
        handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "interrupted")
        cls.signal_code = compile(ast.Module(body=[handler], type_ignores=[]), str(launcher), "exec")

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="v1-launcher-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work = self.root / "work"
        self.archive = self.root / "archive"
        self.work.mkdir()
        (self.archive / "phases/tests").mkdir(parents=True)
        self.source_paths = ["Libs/population_dynamics/" + name + ".py" for name in
                             ("__init__", "v1_source_assets", "test_v1_source_assets",
                              "provenance", "resource_probe")] + ["run_guarded.py"]
        self.rows = []
        for name in self.source_paths:
            target = self.work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# " + name + "\n")
            self.rows.append({"path": name, "sha256": self.digest(target)})
        self.tests = {"status": "PASS", "exit_code": 0,
                      "command": ["/validated/python3", "-B", "-m", "unittest",
                                  "population_dynamics.test_v1_source_assets", "-v"]}
        self.delivery = {"exit_code": 3, "job_id": "archived-job", "files": self.rows}
        self.write_evidence()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_evidence(self):
        receipt = self.archive / "phases/tests/receipt.json"
        receipt.write_text(json.dumps(self.tests))
        self.delivery["files"] = [row for row in self.delivery["files"]
                                  if row["path"] != "phases/tests/receipt.json"]
        self.delivery["files"].append({"path": "phases/tests/receipt.json",
                                       "sha256": self.digest(receipt)})
        self.write_delivery()

    def write_delivery(self):
        (self.archive / "delivery.json").write_text(json.dumps(self.delivery))

    def reuse(self):
        environment = {"NRNSIM_V1_WORK": str(self.work),
                       "NRNSIM_V1_TEST_ARCHIVE": str(self.archive),
                       "NRNSIM_V1_PYTHON": "/validated/python3"}
        with mock.patch.dict(os.environ, environment), contextlib.redirect_stdout(io.StringIO()):
            exec(self.reuse_code, {})

    def test_resource_interrupt_cannot_be_wrapped_as_request_exception(self):
        namespace = {}
        exec(self.signal_code, namespace)
        with self.assertRaises(KeyboardInterrupt) as error:
            namespace["interrupted"](15, None)
        self.assertIn("SOURCE_ASSET_INTERRUPTED_BY_SIGNAL:15", str(error.exception))

    def test_passed_tests_reused_after_acquisition_failure(self):
        self.reuse()
        result = json.loads((self.work / "reused_tests.json").read_text())
        self.assertEqual(result["status"], "PASS_REUSED_NATIVE_TESTS")
        self.assertEqual(result["job_id"], "archived-job")
        self.assertEqual(result["verified_source_files"], self.source_paths)

    def test_passed_tests_reused_after_successful_delivery(self):
        self.delivery["exit_code"] = 0
        self.write_delivery()
        self.reuse()
        self.assertTrue((self.work / "reused_tests.json").exists())

    def test_unsuccessful_test_phase_rejected_even_if_delivery_succeeded(self):
        self.delivery["exit_code"] = 0
        for status, code in (("FAILED", 1), ("RESOURCE_BLOCKED", 1),
                             ("RUNNING", None), ("PASS", 1)):
            with self.subTest(status=status, code=code):
                self.tests.update(status=status, exit_code=code)
                self.write_evidence()
                with self.assertRaisesRegex(ValueError, "V1_PRIOR_NATIVE_TESTS_DID_NOT_PASS"):
                    self.reuse()
                self.assertFalse((self.work / "reused_tests.json").exists())

    def test_changed_test_command_rejected(self):
        self.tests["command"][0] = "/different/python3"
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, "V1_PRIOR_NATIVE_TEST_COMMAND_CHANGED"):
            self.reuse()

    def test_each_changed_source_rejected(self):
        for name in self.source_paths:
            with self.subTest(source=name):
                target = self.work / name
                original = target.read_bytes()
                target.write_bytes(original + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "V1_PRIOR_TESTED_SOURCE_CHANGED"):
                    self.reuse()
                target.write_bytes(original)

    def test_changed_archived_receipt_rejected(self):
        receipt = self.archive / "phases/tests/receipt.json"
        receipt.write_text(receipt.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "V1_PRIOR_TEST_RECEIPT_CHANGED"):
            self.reuse()

    def test_missing_receipt_or_source_manifest_rows_rejected(self):
        for name in ("phases/tests/receipt.json", self.source_paths[1]):
            with self.subTest(missing=name):
                original = self.delivery["files"]
                self.delivery["files"] = [row for row in original if row["path"] != name]
                self.write_delivery()
                with self.assertRaises(KeyError):
                    self.reuse()
                self.delivery["files"] = original


    def test_cpu04_and_cpu05_accept_the_existing_resume_interface(self):
        arguments = ["/repo", "/receipt", "192.0.2.10", "full", "/archive", "/prefix"]
        for node in ("cpu04", "cpu05"):
            with self.subTest(node=node):
                env = dict(os.environ, SLURMD_NODENAME=node, SLURM_JOB_GPUS="")
                result = subprocess.run(["bash", "-s", "--", *arguments],
                    input=self.placement_source, text=True, capture_output=True, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "PLACEMENT_ACCEPTED")

    def test_other_nodes_and_gpu_allocations_are_rejected_before_runtime_setup(self):
        arguments = ["/repo", "/receipt", "192.0.2.10", "full", "/archive", "/prefix"]
        for node, gpus in (("cpu03", ""), ("gpu03", ""), ("", ""),
                           ("cpu04", "0"), ("cpu05", "0")):
            with self.subTest(node=node, gpus=gpus):
                env = dict(os.environ, SLURMD_NODENAME=node, SLURM_JOB_GPUS=gpus)
                result = subprocess.run(["bash", "-s", "--", *arguments],
                    input=self.placement_source, text=True, capture_output=True, env=env)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("PLACEMENT_ACCEPTED", result.stdout)


if __name__ == "__main__":
    unittest.main()
