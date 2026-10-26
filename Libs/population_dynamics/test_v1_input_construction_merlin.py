"""Focused caller contracts for the isolated Merlin input-construction runner."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts import v1_input_construction_merlin as runner
from . import resource_probe


class MerlinInputConstructionTests(unittest.TestCase):
    def recipe(self):
        return dict(schema_version=1, source_root="/original/source", expected_full_nodes=230924,
                    canary_fraction=0.01, full_fraction=1.0, edge_chunk_size=1000000,
                    builder_seed=400000098, glif_assignment_seed=401000098,
                    bmtk_wheel=dict(path="/original/bmtk.whl", sha256="bound-wheel"),
                    benchmark_circuit_candidate=dict(circuit_index=98, circuit_id="v1_visual_circuit_098"))

    def test_resolved_config_preserves_the_exact_original_envelope(self):
        original = self.recipe()
        before = copy.deepcopy(original)
        result = runner.resolved_config(original, "original-config-digest", Path("/isolated/work"))
        self.assertEqual(original, before)
        self.assertEqual(result["original_config"], before)
        self.assertEqual(result["original_config_sha256"], "original-config-digest")
        self.assertEqual(result["source_root"], "/isolated/work/sources")
        self.assertEqual(result["bmtk_wheel"], dict(path="/isolated/work/bmtk.whl", sha256="bound-wheel"))
        self.assertEqual(result["builder_seed"], 400000098)
        self.assertEqual(result["benchmark_circuit_candidate"], before["benchmark_circuit_candidate"])

    def test_recipe_allows_declared_realizations_and_rejects_reduced_scale(self):
        reference = self.recipe()
        candidate = copy.deepcopy(reference)
        candidate.update(builder_seed=400000099, glif_assignment_seed=401000099, source_root="/other/source",
                         benchmark_circuit_candidate=dict(circuit_index=99, circuit_id="v1_visual_circuit_099"))
        candidate["bmtk_wheel"]["path"] = "/other/bmtk.whl"
        runner.require_canary_recipe(candidate, reference)
        candidate["full_fraction"] = 0.01
        with self.assertRaisesRegex(ValueError, "RECIPE_CHANGED_SINCE_CANARY"):
            runner.require_canary_recipe(candidate, reference)

    def test_recipe_rejects_a_different_wheel_identity(self):
        reference = self.recipe()
        candidate = copy.deepcopy(reference)
        candidate["bmtk_wheel"]["sha256"] = "different-wheel"
        with self.assertRaisesRegex(ValueError, "RECIPE_CHANGED_SINCE_CANARY"):
            runner.require_canary_recipe(candidate, reference)

    def test_merlin_resource_probe_preserves_the_original_memory_boundary(self):
        adapted = types.ModuleType("population_dynamics.merlin_resource_probe_fixture")
        adapted.__package__ = "population_dynamics"
        source = Path(resource_probe.__file__).read_text()
        exec(compile(runner.merlin_resource_probe(source), "<merlin-resource-probe>", "exec"), adapted.__dict__)
        environment = dict(RH2_JOB_RUN_ID="fixture-run", ARNOLD_TRIAL_ID="fixture-trial",
                           ARNOLD_MONITOR_ROLE="head", ARNOLD_APPLIED_GPU_NUM="0", ARNOLD_HEAD_CPU="2",
                           CUDA_VISIBLE_DEVICES="")
        telemetry = "procs\nr b swpd free buff cache si so bi bo in cs us sy id wa st\n" + (
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 0 0\n" * 4)
        for available_gib, status in ((64, "PASS"), (63, "BLOCKED")):
            snapshot = dict(d_state_count=0, disk_bytes={"/": dict(total=200 * 1024**3,
                            free=150 * 1024**3, used=50 * 1024**3)},
                            memory_bytes={"MemAvailable": available_gib * 1024**3})
            with self.subTest(available_gib=available_gib), tempfile.TemporaryDirectory() as directory:
                with mock.patch.dict(adapted.os.environ, environment, clear=True), \
                     mock.patch.object(adapted, "snapshot", return_value=snapshot), \
                     mock.patch.object(adapted.subprocess, "check_output", return_value=telemetry):
                    report = adapted.probe(Path(directory) / "resources.json", runner.FULL_REQUIRED_MEMORY_GIB)
                    self.assertEqual(report["status"], status)
                    self.assertEqual(report["execution_context"]["backend"], "merlin")
                    self.assertEqual(report["execution_context"]["run_id"], "fixture-run")
                    self.assertNotIn("slurm_job_id", report)
                    self.assertEqual("INSUFFICIENT_AVAILABLE_MEMORY" in report["reasons"], available_gib < 64)
                    saved = json.loads((Path(directory) / "resources.json").read_text())
                    self.assertEqual(saved["execution_context"], report["execution_context"])


if __name__ == "__main__":
    unittest.main()
