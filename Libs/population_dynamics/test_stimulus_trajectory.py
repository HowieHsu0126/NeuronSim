"""Cohort independence, causal observation clocks and immutable acquisition."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from .nest_gpu_network import ticks
from .stimulus_trajectory import acquire, build_plan, freeze
from .provenance import sha256_file
from .test_pulse_history import design


class TrajectoryCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.graph = self.root / "graph"
        self.graph.mkdir()
        _, graph = design()
        (self.graph / "manifest.json").write_text(json.dumps(graph))
        self.library = self.root / "rng.so"
        self.library.write_bytes(b"source RNG fixture")
        config_path = Path(__file__).parents[1] / "configs/neural_population_stimulus_trajectory.yaml"
        self.config = yaml.safe_load(config_path.read_text())

    def tearDown(self):
        self.temp.cleanup()

    def test_blocks_templates_and_complete_factorial_denominators(self):
        plan = build_plan(self.config, self.graph, self.library)
        self.assertEqual(len(plan["blocks"]), 36)
        self.assertEqual(len(plan["cases"]), 360)
        for block in plan["blocks"]:
            rows = [r for r in plan["cases"] if r["block"] == block["block"]]
            self.assertEqual({r["split"] for r in rows}, {block["split"]})
            self.assertEqual({r["initial_state_seed"] for r in rows}, {block["initial_state_seed"]})
            self.assertEqual({r["background_seed"] for r in rows}, {block["background_seed"]})
            expected = 4 if block["split"] == "train" else 16
            self.assertEqual(len(rows), expected)
            if block["split"] != "train":
                factorial = [r for r in rows if r["family"] == "factorial"]
                self.assertEqual(len({(r["focal_arrival_ms"], r["history"], r["present"]) for r in factorial}), 8)
        familiar = [t for t in plan["templates"] if t["family"] == "familiar"]
        for position in range(9):
            self.assertEqual({(t["motifs"][position]["local_conditioning"],
                               t["motifs"][position]["focal_present"]) for t in familiar},
                             {(False, False), (False, True), (True, False), (True, True)})
        for split in ("train", "calibration", "test"):
            selected = {r["template_id"] for r in plan["cases"] if r["split"] == split and r["family"] == "trajectory"}
            self.assertFalse(any(t.startswith("test_timing") for t in selected) and split != "test")
            self.assertFalse(any(t.startswith("calibration_timing") for t in selected) and split != "calibration")

    def test_snapshots_precede_emission_and_complete_past_controls_are_retained(self):
        plan = build_plan(self.config, self.graph, self.library)
        template_map = {t["template_id"]: t for t in plan["templates"]}
        for row in plan["cases"]:
            settings = row["settings"]
            if row["family"] == "trajectory":
                template = template_map[row["template_id"]]
                focal = [template["motifs"][i]["focal_arrival_ms"] for i in [0, 4, 8]]
                arrivals = [p for motif in template["motifs"] for p in motif["pulse_arrivals_ms"]]
                self.assertEqual(settings["pulse_control"]["initial_emission_ms"], [v - 1 for v in arrivals])
            else:
                focal = [row["focal_arrival_ms"]]
            self.assertEqual(settings["state_observation_ticks"],
                             [int(ticks([v - 1], .1)[0]) - 1 for v in focal])

    def test_reused_seed_or_overlapping_motif_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["cohorts"]["test"]["initial_state_seeds"][0] = 83001
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_plan(config, self.graph, self.library)
        config = copy.deepcopy(self.config)
        config["templates"]["familiar"]["focal_arrivals_ms"][1] = 1300.
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_plan(config, self.graph, self.library)

    def test_freeze_preserves_existing_plan_on_changed_seed_or_timing(self):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(self.config))
        output = self.root / "prospective.json"
        freeze(path, self.graph, self.library, output)
        original = output.read_bytes()
        self.config["template_seed"] += 1
        path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            freeze(path, self.graph, self.library, output)
        self.assertEqual(output.read_bytes(), original)

    def test_native_source_variant_is_separate_from_initial_state_rng(self):
        native = self.root / "native.so"
        native.write_bytes(b"native Philox variant")
        plan = build_plan(self.config, self.graph, self.library, native, "per_edge_philox4_32_10")
        self.assertEqual(plan["source_rng_sha256"], sha256_file(self.library))
        self.assertEqual(plan["native_library_sha256"], sha256_file(native))
        self.assertTrue(all(c["settings"]["expected_poisson_rng_family"] == "per_edge_philox4_32_10"
                            for c in plan["cases"]))
        self.assertNotIn("expected_poisson_rng_family", self.config["execution"])
        with self.assertRaisesRegex(ValueError, "declared Poisson"):
            build_plan(self.config, self.graph, self.library, native)
        path = self.root / "variant.json"
        path.write_text(json.dumps(plan))
        native.write_bytes(b"changed native runtime")
        with patch.dict(os.environ, NESTGPU_LIB=str(native)):
            with self.assertRaisesRegex(ValueError, "native runtime changed"):
                acquire(path, "train", self.graph, self.library, self.root / "output")
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
