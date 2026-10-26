"""Check launcher handling of a child that cannot write a Python exception."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import prepare_population_residual_fitting_merlin as preparer
from unittest.mock import Mock, patch

from prepare_population_residual_fitting_merlin import BOOTSTRAP, DEFAULT_CONFIG, MEAN_CONFIG, validated_source_config


class LauncherExitTests(unittest.TestCase):
    def launch(self, directory, code, config_relative=None, threads=None):
        root = Path(directory)
        wrapper = b"# test publisher provided through sys.modules\n"
        manifest = b"{}"
        publisher = types.ModuleType("publisher")

        def stage(source, archive, work):
            (work / "inputs").mkdir()
            (work / "inputs/manifest.json").write_bytes(manifest)
            (work / "inputs.tar").write_bytes(b"archive")
            return {"staged": True}

        publisher.stage = stage
        publisher.fingerprint = lambda path: dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        publisher.verify = Mock()
        publisher.persist = Mock(return_value={"persisted": True, "exit_code": code})
        process = Mock()
        process.pid = 123
        process.poll.return_value = code
        payload = dict(publisher=dict(hdfs_path="test://publisher", bytes=len(wrapper),
                                     sha256=hashlib.sha256(wrapper).hexdigest()),
                       archive=dict(bytes=1, sha256="archive"), source={}, restore=[],
                       input_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                       phase="qualify", device="cpu", seeds=[41], modes=["current"], epoch_limit=None)

        if config_relative is not None:
            payload.update(config_relative=config_relative, config_identity={"bytes": 10, "sha256": "config"})

        if threads is not None:
            payload["threads"] = threads

        def fetch(command, **kwargs):
            self.assertEqual(command[:3], ["hdfs", "dfs", "-get"])
            Path(command[-1]).write_bytes(wrapper)

        with patch.dict(sys.modules, {"publisher": publisher}), \
                patch.dict(os.environ, RH2_JOB_RUN_ID="test_run", ARNOLD_TRIAL_ID="test_trial",
                           RH2_CHECKPOINT_HDFS="test://checkpoints"), \
                patch("tempfile.mkdtemp", return_value=directory), \
                patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=20 * 1024**3)), \
                patch("subprocess.run", side_effect=fetch), \
                patch("subprocess.Popen", return_value=process) as popen:
            with self.assertRaises(SystemExit) as stopped:
                exec(compile(BOOTSTRAP, "fitting_bootstrap", "exec"), {"payload": payload})
            self.assertEqual(stopped.exception.code, code)
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONFAULTHANDLER"], "1")
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--threads") + 1], str(threads or 8))
            self.assertEqual(command[command.index("--config") + 1], str(root / "source" / (config_relative or DEFAULT_CONFIG)))
            if config_relative is not None:
                publisher.verify.assert_called_once_with(root / "source" / config_relative, payload["config_identity"])
        sys.path.remove(directory)
        return publisher, root / "results"

    def test_signal_failure_is_persisted_without_claiming_fit_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher, output = self.launch(directory, -11)
            report = json.loads((output / "execution.json").read_text())
            self.assertEqual(report["status"], "FAILED_FIT_PROCESS")
            self.assertEqual(report["child_exit_code"], -11)
            self.assertFalse(report["fitting_execution_receipt_available"])
            self.assertFalse(report["scientific_claim_qualified"])
            self.assertEqual(json.loads((output / "analysis.json").read_text()), report)
            self.assertTrue((output / "fitting.log").is_file())
            self.assertEqual(publisher.persist.call_args.args[-2:], ("test://checkpoints/fit_launcher", -11))

    def test_mean_launch_passes_and_verifies_explicit_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self.launch(directory, 0, MEAN_CONFIG)

    def test_selected_threads_reach_worker_command(self):
        with tempfile.TemporaryDirectory() as directory:
            self.launch(directory, 0, MEAN_CONFIG, threads=2)

    def test_success_does_not_create_a_failure_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher, output = self.launch(directory, 0)
            publisher.persist.assert_not_called()
            self.assertFalse((output / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()


class SourceConfigurationTests(unittest.TestCase):
    def fixture(self, root):
        sources = {name: dict(bytes=10, sha256="config") for name in (DEFAULT_CONFIG, MEAN_CONFIG)}
        for name in ("population_residual_mean_objective.py", "native_renewal_expectation.py",
                     "population_residual_observation_fitting.py", "population_residual_observation_training.py",
                     "population_residual_observation_execution.py", "poisson_trajectory_evaluation.py"):
            sources["population_dynamics/" + name] = dict(bytes=10, sha256="source")
        result = dict(status="PASS", exit_code=0, tests=20, failures=0, errors=0, skipped=0,
                      source_payload_sha256="payload", sources=sources)
        upload = dict(complete=True, bytes=100, sha256="payload", hdfs_path="test://source")
        (root / "result.json").write_text(json.dumps(result))
        (root / "source_upload.json").write_text(json.dumps(upload))
        return result, upload

    def test_explicit_mean_config_selects_same_tested_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, upload = self.fixture(root)
            actual, identity = validated_source_config(root, MEAN_CONFIG)
            self.assertEqual(actual, upload)
            self.assertEqual(identity, result["sources"][MEAN_CONFIG])
            self.assertEqual(validated_source_config(root, DEFAULT_CONFIG)[0], upload)

    def test_unqualified_missing_or_changed_config_is_rejected(self):
        for failure in ("failed", "payload", "config", "module"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result, _ = self.fixture(root)
                if failure == "failed":
                    result["failures"] = 1
                elif failure == "payload":
                    result["source_payload_sha256"] = "changed"
                elif failure == "config":
                    del result["sources"][MEAN_CONFIG]
                else:
                    del result["sources"]["population_dynamics/native_renewal_expectation.py"]
                (root / "result.json").write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    validated_source_config(root, MEAN_CONFIG)

    def test_config_path_cannot_escape_or_select_unregistered_source(self):
        with tempfile.TemporaryDirectory() as folder:
            for relative in ("../config.yaml", "/tmp/config.yaml", "configs/other.yaml"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    validated_source_config(Path(folder), relative)


class RequestResourceTests(unittest.TestCase):
    fixture = SourceConfigurationTests.fixture
    def prepare(self, root, device, threads=8):
        checks = root / "checks"
        checks.mkdir()
        self.fixture(checks)
        request = dict(job_config=dict(job_template_config=dict(env_map={"CUDA_VISIBLE_DEVICES": "", "KEEP": "value"})),
            resource_config=dict(arnold_resource_config=dict(cluster_ids=[20], group_id=271,
                roles=[dict(name="head", queue_name="cpu", cpu=4, memory_mb=8192, gpu_type="CPU_ONLY")])))
        (checks / "request.json").write_text(json.dumps(request))
        manifest = root / "native_observation_fitting/real_inputs/full_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        admission = root / "admission.json"
        admission.write_text(json.dumps(dict(status="VERIFIED_INPUT_BUNDLE", exit_code=0, logical_files=462,
            input_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), archive=dict(bytes=1, sha256="archive", hdfs_path="test://input"))))
        publisher = root / "merlin_readout_analysis/wrapper_upload.json"
        publisher.parent.mkdir()
        publisher.write_text(json.dumps(dict(sha256="publisher", bytes=1, hdfs_path="test://publisher")))
        output = root / "prepared"
        argv = ["prepare", "--campaign-root", str(root), "--admission", str(admission), "--output", str(output),
            "--source-checks", str(checks), "--config-relative", MEAN_CONFIG, "--name", "mrln_test_mean_config",
            "--phase", "qualify", "--seeds", "41", "--device", device, "--threads", str(threads)]
        if device == "cuda":
            argv += ["--cluster-id", "34", "--group-id", "2088", "--queue", "gpu-queue", "--gpu-type", "A100_SXM_80GB"]
        with patch.object(sys, "argv", argv):
            preparer.main()
        return json.loads((output / "request.json").read_text())

    def test_gpu_request_removes_cpu_visibility_mask_and_preserves_allocated_role(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self.prepare(Path(folder), "cuda", threads=2)
        environment = request["job_config"]["job_template_config"]["env_map"]
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["KEEP"], "value")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertEqual(environment[name], "2")
        role = request["resource_config"]["arnold_resource_config"]["roles"][0]
        self.assertEqual((role["gpu"], role["gpu_type"], role["queue_name"]), (1, "A100_SXM_80GB", "gpu-queue"))
        self.assertEqual(request["resource_config"]["arnold_resource_config"]["cluster_ids"], [34])
        self.assertNotIn("export CUDA_VISIBLE_DEVICES=''", request["job_config"]["job_template_config"]["entrypoint_full_script"])

    def test_cpu_request_retains_cpu_only_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self.prepare(Path(folder), "cpu")
        self.assertEqual(request["job_config"]["job_template_config"]["env_map"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("export CUDA_VISIBLE_DEVICES=''", request["job_config"]["job_template_config"]["entrypoint_full_script"])
        self.assertNotIn("gpu", request["resource_config"]["arnold_resource_config"]["roles"][0])
