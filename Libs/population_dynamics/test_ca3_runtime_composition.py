"""Small-file contract tests for explicit CA3 backend composition.

These fixtures do not compile or execute a simulator and draw no random samples.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import ca3_benchmark as benchmark

LIBRARY = "libcarlsim.so.6.0.0"
STATUS = "BUILT_UNQUALIFIED_CA3_SM80_BACKEND"


def file_identity(path: Path) -> dict:
    payload = path.read_bytes()
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


class TestCA3RuntimeComposition(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / "parent"
        self.runtime = self.root / "candidate"
        (self.parent / "lib").mkdir(parents=True)
        (self.parent / "bin").mkdir()
        for name, content in ((LIBRARY, b"original backend"), ("libcurand.so.10", b"original curand"),
                              ("libcarlsim-stopwatch.so", b"original stopwatch")):
            (self.parent / "lib" / name).write_bytes(content)
        (self.parent / "bin/ca3_benchmark").write_bytes(b"unchanged driver")
        (self.parent / "trainPat.csv").write_bytes(b"1 0 1\n")
        self.original_libraries = {path.name: file_identity(path)["sha256"]
                                   for path in (self.parent / "lib").iterdir()}
        pattern = file_identity(self.parent / "trainPat.csv")["sha256"]
        self.built = dict(
            status="BUILT_CA3_BENCHMARK_NATIVE_CONTROLS",
            binary_sha256=file_identity(self.parent / "bin/ca3_benchmark")["sha256"],
            control_program=dict(duration_ms=1210), pattern_sha256=pattern,
            parameter_identity="fixture-parameter", backend_libraries=self.original_libraries,
            backend_build_report=dict(framework_revision="fixture-framework", pattern_sha256=pattern,
                                      source_file_byte_checks=[{}, dict(staged_sha256="fixture-parameter")]))
        self.parent_report = json.dumps(self.built, indent=2).encode() + b"\n"
        (self.parent / "build_report.json").write_bytes(self.parent_report)
        shutil.copytree(self.parent, self.runtime)
        (self.runtime / "lib" / LIBRARY).write_bytes(b"candidate backend")
        self.library = dict(name=LIBRARY, **file_identity(self.runtime / "lib" / LIBRARY))
        self.build_report = dict(
            status=STATUS, candidate_qualified=False, scientific_fidelity_qualified=False,
            framework_revision="fixture-framework", original_library_sha256=self.original_libraries[LIBRARY],
            original_curand_sha256=self.original_libraries["libcurand.so.10"],
            cuda_architectures=["sm_80"],
            library=dict(path="lib/" + LIBRARY, **file_identity(self.runtime / "lib" / LIBRARY)))
        self.receipt = dict(status=STATUS, exit_code=0, persisted=True,
                            candidate_library=deepcopy(self.build_report["library"]),
                            merlin_job_run_id="fixture-recovery", merlin_trial_id="fixture-trial",
                            builds_run=0, original_build_exit_code=1)
        self.composition = dict(
            runtime_root=str(self.runtime),
            parent_runtime=dict(root=str(self.parent),
                                build_report_sha256=file_identity(self.parent / "build_report.json")["sha256"]),
            candidate_library=self.library, candidate_build_report={}, candidate_receipt={})
        self.descriptor = self.root / "runtime_composition.json"
        self.write_evidence()

    def write_evidence(self):
        for key, document in (("candidate_build_report", self.build_report),
                              ("candidate_receipt", self.receipt)):
            path = self.root / (key + ".json")
            path.write_text(json.dumps(document, indent=2) + "\n")
            self.composition[key] = dict(path=str(path), **file_identity(path))
        self.write_descriptor()

    def write_descriptor(self):
        self.descriptor.write_text(json.dumps(self.composition, indent=2) + "\n")

    def validate(self):
        return benchmark.validate_runtime_composition(self.runtime, self.built, self.descriptor)

    def test_default_original_runtime_is_compatible(self):
        libraries, composition = benchmark.validate_runtime_composition(self.parent, self.built)
        self.assertEqual(libraries, self.original_libraries)
        self.assertIsNone(composition)
        self.assertEqual((self.parent / "build_report.json").read_bytes(), self.parent_report)

    def test_candidate_requires_explicit_composition(self):
        with self.assertRaisesRegex(ValueError, "CA3_BENCHMARK_RUNTIME_LIBRARY_CHANGED"):
            benchmark.validate_runtime_composition(self.runtime, self.built)

    def test_explicit_candidate_preserves_parent_and_binds_recovery(self):
        libraries, composition = self.validate()
        expected = dict(self.original_libraries)
        expected[LIBRARY] = self.library["sha256"]
        self.assertEqual(libraries, expected)
        self.assertEqual(self.built["backend_libraries"], self.original_libraries)
        self.assertEqual((self.parent / "build_report.json").read_bytes(), self.parent_report)
        self.assertEqual((self.runtime / "build_report.json").read_bytes(), self.parent_report)
        self.assertEqual(composition["candidate_receipt"], self.composition["candidate_receipt"])
        self.assertEqual(composition["candidate_receipt_execution"]["merlin_job_run_id"], "fixture-recovery")
        self.assertFalse(composition["candidate_qualified"])
        self.assertFalse(composition["scientific_fidelity_qualified"])

    def test_parent_root_must_be_distinct(self):
        self.composition["parent_runtime"]["root"] = str(self.runtime)
        self.write_descriptor()
        with self.assertRaisesRegex(ValueError, "DISTINCT_ROOTS_REQUIRED"):
            self.validate()

    def test_parent_report_hash_mismatch_is_rejected(self):
        self.composition["parent_runtime"]["build_report_sha256"] = "0" * 64
        self.write_descriptor()
        with self.assertRaisesRegex(ValueError, "PARENT_REPORT_CHANGED"):
            self.validate()

    def test_copied_parent_report_must_be_byte_identical(self):
        (self.runtime / "build_report.json").write_text(json.dumps(self.built))
        with self.assertRaisesRegex(ValueError, "PARENT_REPORT_CHANGED"):
            self.validate()

    def test_changed_parent_backend_is_rejected(self):
        (self.parent / "lib" / LIBRARY).write_bytes(b"changed original backend")
        with self.assertRaisesRegex(ValueError, "PARENT_LIBRARY_CHANGED"):
            self.validate()

    def test_changed_candidate_file_is_rejected(self):
        path = self.runtime / "lib" / LIBRARY
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "CA3_BENCHMARK_RUNTIME_LIBRARY_CHANGED"):
            self.validate()

    def test_candidate_library_name_and_size_are_bound(self):
        original = deepcopy(self.composition)
        for key, value in (("name", "libcurand.so.10"), ("bytes", self.library["bytes"] + 1)):
            with self.subTest(field=key):
                self.composition = deepcopy(original)
                self.composition["candidate_library"][key] = value
                self.write_descriptor()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_build_report_and_receipt_bytes_are_bound(self):
        for key in ("candidate_build_report", "candidate_receipt"):
            with self.subTest(artifact=key):
                path = Path(self.composition[key]["path"])
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "ARTIFACT_CHANGED:" + key):
                    self.validate()
                path.write_bytes(original)

    def test_candidate_report_semantics_are_required(self):
        original = deepcopy(self.build_report)
        changes = (("status", "BUILT_AND_QUALIFIED"), ("candidate_qualified", True),
                   ("scientific_fidelity_qualified", True), ("framework_revision", "other-framework"),
                   ("original_library_sha256", "0" * 64), ("original_curand_sha256", "0" * 64),
                   ("cuda_architectures", ["sm_89"]),
                   ("library", dict(path="lib/" + LIBRARY, bytes=1, sha256="0" * 64)))
        for key, value in changes:
            with self.subTest(field=key):
                self.build_report = deepcopy(original)
                self.build_report[key] = value
                self.write_evidence()
                with self.assertRaisesRegex(ValueError, "CANDIDATE_REPORT_DIFFERS"):
                    self.validate()

    def test_receipt_requires_success_persistence_and_candidate_identity(self):
        original = deepcopy(self.receipt)
        for key, value in (("exit_code", 1), ("exit_code", False), ("persisted", False),
                           ("candidate_library", dict(bytes=1, sha256="0" * 64))):
            with self.subTest(field=key, value=value):
                self.receipt = deepcopy(original)
                self.receipt[key] = value
                self.write_evidence()
                with self.assertRaisesRegex(ValueError, "SUCCESSFUL_RECEIPT_REQUIRED"):
                    self.validate()

    def test_non_candidate_libraries_keep_original_hashes(self):
        for name in ("libcurand.so.10", "libcarlsim-stopwatch.so"):
            with self.subTest(library=name):
                path = self.runtime / "lib" / name
                original = path.read_bytes()
                path.write_bytes(b"changed noncandidate library")
                with self.assertRaisesRegex(ValueError, "CA3_BENCHMARK_RUNTIME_LIBRARY_CHANGED"):
                    self.validate()
                path.write_bytes(original)

    def test_cli_forwards_optional_composition(self):
        base = ["ca3_benchmark", "run", "--packaged", str(self.runtime),
                "--formation", "formation", "--analysis", "analysis", "--config", "config",
                "--output", "output", "--block-index", "0", "--condition-index", "3"]
        for descriptor in (None, self.descriptor):
            with self.subTest(composition=descriptor):
                argv = base if descriptor is None else base + ["--runtime-composition", str(descriptor)]
                with patch("sys.argv", argv), patch.object(benchmark, "run") as run:
                    benchmark.main()
                self.assertEqual(run.call_args.kwargs["runtime_composition"], descriptor)

    def test_episode_report_discloses_candidate_before_native_launch(self):
        formation, analysis = self.root / "formation", self.root / "analysis"
        formation.mkdir()
        analysis.mkdir()
        (formation / "native_report.json").write_text("{}")
        normalized = analysis / "network65p_assem550_divnorm.dat"
        normalized.write_bytes(b"fixture normalized input")
        binding = dict(normalized_input_sha256=file_identity(normalized)["sha256"])
        protocol = dict(retrieval_duration_ms=1210)
        campaign = dict(sampling=dict(circuit_instances=1),
                        scenarios=dict(ca3_pattern_completion=dict(task_protocol=protocol)))
        # Literal stream identifiers are never used to draw samples in this test.
        identity = dict(circuit_index=0, block_index=0, condition_index=3,
                        stimulus_seed=0, background_seed=0, initial_state_seed=0)
        for descriptor, runtime in ((None, self.parent), (self.descriptor, self.runtime)):
            with self.subTest(composition=descriptor):
                output = self.root / ("original_episode" if descriptor is None else "candidate_episode")
                with patch.dict(os.environ, SLURM_JOB_ID="merlin_fixture", CUDA_VISIBLE_DEVICES="0"), \
                        patch.multiple(benchmark,
                            campaign_config=Mock(return_value=campaign),
                            episode_identities=Mock(return_value=[identity]),
                            condition_for_episode=Mock(return_value=dict(condition_id="completion_50")),
                            qualify_input=Mock(return_value=binding),
                            probe=Mock(return_value=dict(status="PASS")),
                            ca3_cue_targets=Mock(return_value=[]),
                            native_launch_command=Mock(return_value=["fixture_native"])), \
                        patch.object(benchmark.subprocess, "Popen", side_effect=RuntimeError("fixture launch stopped")):
                    with self.assertRaisesRegex(RuntimeError, "fixture launch stopped"):
                        benchmark.run(runtime, formation, analysis, self.root / "config", output,
                                      block_index=0, condition_index=3, runtime_composition=descriptor)
                report = json.loads((output / "report.json").read_text())
                self.assertFalse(report["scientific_fidelity_qualified"])
                self.assertEqual(report["production_episodes"], 0)
                if descriptor is None:
                    self.assertNotIn("runtime_composition", report)
                    self.assertNotIn("effective_runtime_libraries", report)
                else:
                    self.assertEqual(report["runtime_composition"]["parent_runtime"],
                                     self.composition["parent_runtime"])
                    self.assertEqual(report["effective_runtime_libraries"][LIBRARY], self.library["sha256"])
                    self.assertFalse(report["runtime_composition"]["candidate_qualified"])


if __name__ == "__main__":
    unittest.main()
