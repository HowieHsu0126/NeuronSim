"""Source comparisons preserve blocks, controls and native-runtime identity."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from .provenance import sha256_file
from .source_variant_cohort import acquire, declare, verify_episode
from .stimulus_trajectory import build_plan
from .test_pulse_history import design


class SourceVariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "graph"
        self.graph.mkdir()
        _, graph = design()
        (self.graph / "manifest.json").write_text(json.dumps(graph))
        self.rng = self.root / "rng.so"
        self.rng.write_bytes(b"initial state RNG")
        self.native = self.root / "candidate.so"
        self.native.write_bytes(b"candidate native runtime")
        configs = Path(__file__).parents[1] / "configs"
        self.config = yaml.safe_load((configs / "neural_population_rng_neural_comparison.yaml").read_text())
        source_config = yaml.safe_load((configs / "neural_population_stimulus_trajectory.yaml").read_text())
        self.plan = build_plan(source_config, self.graph, self.rng)
        self.original = self.root / "original.json"
        self.original.write_text(json.dumps(self.plan))
        self.reference = self.root / "reference"
        self.reference.mkdir()
        (self.reference / "design.json").write_text(json.dumps(
            {"prospective_design_sha256": sha256_file(self.original)}))
        (self.reference / "acquisition.json").write_text(json.dumps({"status": "ACQUIRED"}))
        self.selected = [c for c in self.plan["cases"]
                         if c["split"] == "train" and c["template_id"] == "familiar_0"]
        for case in self.selected:
            folder = self.reference / case["case_id"]
            folder.mkdir()
            (folder / "report.json").write_text(json.dumps(dict(
                status="EXECUTED", config=case["settings"],
                graph_manifest_sha256=self.plan["graph_manifest_sha256"],
                native_library_sha256="original library")))

    def declaration(self):
        return declare(self.config, self.original, self.reference, self.native)

    def test_complete_training_blocks_only_and_one_changed_setting(self):
        result = self.declaration()
        self.assertEqual([c["block"] for c in result["cases"]], list(range(18)))
        self.assertFalse(result["common_noise_across_rng_families"])
        self.assertEqual(result["native_library_sha256"], sha256_file(self.native))
        source = {c["case_id"]: c for c in self.selected}
        for case in result["cases"]:
            settings = dict(case["settings"])
            self.assertEqual(settings.pop("expected_poisson_rng_family"), "per_edge_philox4_32_10")
            self.assertEqual(settings, source[case["case_id"]]["settings"])

    def test_missing_block_mixed_reference_and_test_access_are_rejected(self):
        config = copy.deepcopy(self.config)
        config["expected_blocks"] = 17
        with self.assertRaisesRegex(ValueError, "every independent"):
            declare(config, self.original, self.reference, self.native)
        config["split"] = "test"
        with self.assertRaisesRegex(ValueError, "training-only"):
            declare(config, self.original, self.reference, self.native)
        path = self.reference / self.selected[0]["case_id"] / "report.json"
        report = json.loads(path.read_text())
        report["native_library_sha256"] = "another native library"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "mix native"):
            self.declaration()

    def test_runtime_mismatch_stops_before_output_or_execution(self):
        result = self.declaration()
        path = self.root / "comparison.json"
        path.write_text(json.dumps(result))
        self.native.write_bytes(b"changed")
        with patch.dict(os.environ, NESTGPU_LIB=str(self.native)):
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                acquire(path, self.graph, self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_reuse_checks_native_identity_and_artifact_content(self):
        result = self.declaration()
        case = result["cases"][0]
        folder = self.root / "candidate_episode"
        folder.mkdir()
        (folder / "spikes.npz").write_bytes(b"source-bound fixture")
        report = dict(status="EXECUTED", config=case["settings"], checks={"clock": True},
                      graph_manifest_sha256=result["graph_manifest_sha256"],
                      initial_state_rng_library_sha256=result["source_rng_sha256"],
                      native_library_sha256=result["native_library_sha256"],
                      native_poisson_rng_family=result["config"]["candidate_rng_family"],
                      artifacts={"spikes.npz": sha256_file(folder / "spikes.npz")})
        (folder / "report.json").write_text(json.dumps(report))
        verify_episode(folder, case, result)
        report["native_library_sha256"] = "wrong"
        (folder / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "different source"):
            verify_episode(folder, case, result)
        report["native_library_sha256"] = result["native_library_sha256"]
        (folder / "report.json").write_text(json.dumps(report))
        (folder / "spikes.npz").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verify_episode(folder, case, result)


if __name__ == "__main__":
    unittest.main()
