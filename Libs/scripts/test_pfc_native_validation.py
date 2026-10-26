"""Bounded fixtures for native execution gates and authenticated source staging."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_pfc_native_validation as runner


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="neuronsim-pfc-runner-test-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original_path = list(sys.path)
        self.addCleanup(sys.path.__setitem__, slice(None), self.original_path)
        self.project = self.root / "project"
        package = self.project / "Libs/population_dynamics"
        package.mkdir(parents=True)
        for name in runner.MODULES:
            (package / (name + ".py")).write_text('"""Inert test module."""\n')
        script = self.project / "Libs/scripts/run_pfc_source_cases.py"
        script.parent.mkdir(parents=True)
        script.write_text('"""Inert test script."""\n')
        self.detector = self.root / "delivered_detector.py"
        self.detector.write_text(
            "import json\nfrom pathlib import Path\n"
            "def detect_all_resources(output):\n"
            "    Path(output).write_text(json.dumps({'detector': 'delivered-fixture'}))\n"
        )
        self.native_output = self.root / "native_reference"
        self.native_command = [
            "fixture-python", "-B", "-m", "population_dynamics.pfc_native_reference_execution",
            "--output", str(self.native_output), "--execute",
        ]
        self.request = self.root / "request.json"
        self.request.write_text(json.dumps({
            "project_root": str(self.project),
            "native_command": self.native_command,
            "comparison_command": ["fixture-comparison"],
            "comparison_output": str(self.root / "comparison/report.json"),
        }))
        self.resource_calls = []

    def run_fixture(self, *, resource_status="PASS", preflight_exit=0, preflight_only=False,
                    scientific_process=None):
        package = ModuleType("population_dynamics")
        package.__path__ = []
        resource_module = ModuleType("population_dynamics.resource_probe")

        def probe(output, memory_gib):
            self.resource_calls.append((Path(output), memory_gib))
            resources = {"status": resource_status, "reasons": [] if resource_status == "PASS" else ["fixture resource rejection"]}
            Path(output).write_text(json.dumps(resources))
            return resources

        resource_module.probe = probe
        package.resource_probe = resource_module
        argv = ["run_pfc_native_validation", "--request", str(self.request),
                "--resource-detector", str(self.detector)]
        if preflight_only:
            argv.append("--preflight-only")
        with patch.dict(sys.modules, {"population_dynamics": package,
                                     "population_dynamics.resource_probe": resource_module}), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "fixture", "SLURMD_NODENAME": "cpu03",
                                     "SLURM_JOB_GPUS": ""}), \
             patch.object(sys, "argv", argv), \
             patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=32 * 1024**3)), \
              patch.object(runner.subprocess, "run", side_effect=scientific_process,
                           return_value=SimpleNamespace(returncode=preflight_exit)) as process, \
             redirect_stdout(io.StringIO()):
            code = runner.main()
        output = self.root / "run_fixture"
        record = json.loads((output / "execution.json").read_text())
        return code, record, output, process

    def write_archive(self, archive, files, **metadata):
        rows = []
        for name, content in files.items():
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            rows.append({"path": name, "bytes": target.stat().st_size,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        (archive / "delivery.json").write_text(json.dumps(dict(metadata, files=rows)))

    def prepare_comparison(self, source_count):
        self.canonical_config = "canonical_fixture: true\n"
        self.source_archives = []
        for index in range(source_count):
            archive = self.root / ("candidate_" + str(index + 1))
            self.write_archive(archive, {
                "source_run/config.yaml": self.canonical_config,
                "source_run/report.json": json.dumps({"fixture_case": index + 1}),
            })
            self.source_archives.append(archive)
        self.source_inputs = [archive / "source_run" for archive in self.source_archives]
        if source_count > 1:
            self.source_inputs[1] = self.source_archives[1]
        proof = self.root / "factorization_proof"
        for name in ("source_comparison.json", "factorized/report.json", "design.json",
                     "Libs/population_dynamics/pfc_factorized_source.py"):
            target = proof / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("inert factorization fixture\n")
        design = self.root / "preparation/design.json"
        design.parent.mkdir()
        design.write_text("{}\n")
        (design.parent / "native_config.yaml").write_text("native_fixture: true\n")
        command = ["fixture-comparison", "--reference", str(self.native_output / "native")]
        for source in self.source_inputs:
            command += ["--source-run", str(source)]
        self.comparison_output = self.root / "comparison/report.json"
        command += ["--config", str(self.source_archives[0] / "source_run/config.yaml"),
                    "--factorization-proof", str(proof), "--design", str(design),
                    "--output", str(self.comparison_output)]
        self.request.write_text(json.dumps({
            "project_root": str(self.project), "native_command": self.native_command,
            "comparison_command": command, "comparison_output": str(self.comparison_output),
        }))
        self.staged_sources = []

    def complete_scientific_fixture(self, command, **kwargs):
        if "--preflight-only" in command:
            return SimpleNamespace(returncode=0)
        if "--execute" in command:
            self.write_archive(self.native_output, {"native/report.json": "inert native fixture\n"},
                               status="NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE", exit_code=0)
            return SimpleNamespace(returncode=0)
        self.assertEqual(command[0], "fixture-comparison")
        self.staged_sources = [Path(command[index + 1]) for index, argument in enumerate(command)
                               if argument == "--source-run"]
        self.assertEqual(len(self.staged_sources), len(self.source_inputs))
        self.assertEqual(len(set(self.staged_sources)), len(self.source_inputs))
        for index, staged in enumerate(self.staged_sources):
            self.assertTrue(staged.is_relative_to(Path(kwargs["cwd"])))
            self.assertTrue((staged.parent / "delivery.json").is_file())
            self.assertEqual(json.loads((staged / "report.json").read_text()),
                             {"fixture_case": index + 1})
            self.assertEqual((staged / "config.yaml").read_text(), self.canonical_config)
        self.assertEqual(self.staged_sources[0].parent.name, "candidate_source")
        config = Path(command[command.index("--config") + 1])
        self.assertEqual(config, self.staged_sources[0] / "config.yaml")
        self.assertEqual(config.read_text(), self.canonical_config)
        reference = Path(command[command.index("--reference") + 1])
        self.assertEqual((reference / "report.json").read_text(), "inert native fixture\n")
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"status": "FIXTURE_COMPARISON_COMPLETE"}))
        return SimpleNamespace(returncode=0)

    def assert_complete_comparison(self, code, record, process):
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "COMPLETE_AUTHORIZED_NATIVE_AND_COMPARISON")
        self.assertEqual(record["native_exit_code"], 0)
        self.assertEqual(record["comparison_exit_code"], 0)
        self.assertEqual(process.call_count, 3)
        self.assertEqual([phase["name"] for phase in record["phases"]],
                         ["native_preflight", "native_validation", "source_comparison"])
        self.assertEqual(json.loads(self.comparison_output.read_text()),
                         {"status": "FIXTURE_COMPARISON_COMPLETE"})
        origins = record["input_origins"]
        self.assertEqual(origins["candidate"], str(self.source_archives[0]))
        self.assertEqual([row["source_run"] for row in origins["candidates"]],
                         [str(source) for source in self.source_inputs])
        self.assertEqual([row["archive"] for row in origins["candidates"]],
                         [str(archive) for archive in self.source_archives])
        self.assertEqual([row["staged_source_run"] for row in origins["candidates"]],
                         [str(source) for source in self.staged_sources])

    def test_complete_single_source_comparison_preserves_config(self):
        self.prepare_comparison(1)
        code, record, _, process = self.run_fixture(scientific_process=self.complete_scientific_fixture)
        self.assert_complete_comparison(code, record, process)

    def test_complete_multi_source_comparison_stages_every_archive(self):
        self.prepare_comparison(2)
        code, record, _, process = self.run_fixture(scientific_process=self.complete_scientific_fixture)
        self.assert_complete_comparison(code, record, process)

    def test_corruption_in_second_source_archive_closes_comparison(self):
        self.prepare_comparison(2)
        corrupted = self.source_archives[1] / "source_run/report.json"
        corrupted.write_text("corrupted second fixture archive\n")
        code, record, _, process = self.run_fixture(scientific_process=self.complete_scientific_fixture)
        self.assertNotEqual(code, 0)
        self.assertEqual(record["status"], "AUTHORIZED_VALIDATION_FAILED")
        self.assertEqual(record["error_type"], "ValueError")
        self.assertEqual(record["error"], "Declared staging input differs: " + str(corrupted))
        self.assertEqual(process.call_count, 2)
        self.assertEqual([phase["name"] for phase in record["phases"]],
                         ["native_preflight", "native_validation"])
        self.assertEqual(record["native_exit_code"], 0)
        self.assertIsNone(record["comparison_exit_code"])
        self.assertEqual([row["archive"] for row in record["input_origins"]["candidates"]],
                         [str(archive) for archive in self.source_archives])
        self.assertFalse(self.comparison_output.exists())
        self.assertEqual(self.staged_sources, [])

    def test_delivered_detector_and_preflight_only_success(self):
        code, record, output, process = self.run_fixture(preflight_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "PREFLIGHT_COMPLETE_NATIVE_SIMULATION_NOT_EXECUTED")
        self.assertEqual(json.loads((output / "resources.json").read_text()),
                         {"detector": "delivered-fixture"})
        self.assertEqual(record["resource_detector"]["path"], str(self.detector))
        self.assertEqual(record["resource_detector"]["sha256"],
                         hashlib.sha256(self.detector.read_bytes()).hexdigest())
        self.assertEqual(len(self.resource_calls), 1)
        process.assert_called_once()
        command = process.call_args.args[0]
        self.assertIn("--preflight-only", command)
        self.assertNotIn("--execute", command)
        self.assertIsNone(record["native_exit_code"])
        self.assertIsNone(record["comparison_exit_code"])
        self.assertFalse(self.native_output.exists())

    def test_resource_rejection_closes_native_preflight_and_execution(self):
        code, record, _, process = self.run_fixture(resource_status="FAIL")
        self.assertNotEqual(code, 0)
        self.assertEqual(record["status"], "AUTHORIZED_VALIDATION_FAILED")
        self.assertIn("fixture resource rejection", record["error"])
        self.assertEqual(record["phases"], [])
        process.assert_not_called()
        self.assertIsNone(record["native_exit_code"])
        self.assertFalse(self.native_output.exists())

    def test_native_preflight_failure_closes_simulation_and_comparison(self):
        code, record, _, process = self.run_fixture(preflight_exit=7)
        self.assertNotEqual(code, 0)
        self.assertEqual(record["status"], "AUTHORIZED_VALIDATION_FAILED")
        self.assertEqual(record["error"], "Native preflight failed: 7")
        process.assert_called_once()
        self.assertEqual(process.call_args.args[0],
                         self.native_command[:-1] + ["--preflight-only"])
        self.assertEqual([(phase["name"], phase["exit_code"]) for phase in record["phases"]],
                         [("native_preflight", 7)])
        self.assertIsNone(record["native_exit_code"])
        self.assertIsNone(record["comparison_exit_code"])
        self.assertFalse(self.native_output.exists())


if __name__ == "__main__":
    unittest.main()
