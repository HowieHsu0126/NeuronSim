"""Whole-episode identities, disjoint streams, and scientific production gates."""
import collections
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .circuit_campaign import campaign_config, describe, episode_identities, production_prerequisites, freeze_identities


CONFIG = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"


class CircuitCampaignTests(unittest.TestCase):
    def test_actual_identity_denominators_and_whole_circuit_split(self):
        config = campaign_config(CONFIG)
        self.assertEqual(describe(config)["planned_episodes_total"], 300000)
        all_streams = []
        for scenario in config["scenarios"]:
            rows = list(episode_identities(config, scenario))
            self.assertEqual(len(rows), 100000)
            self.assertEqual(len({r["episode_id"] for r in rows}), 100000)
            self.assertEqual(collections.Counter(r["split"] for r in rows),
                             dict(train=70000, validation=10000, calibration=10000, test=10000))
            circuit_splits = collections.defaultdict(set)
            for row in rows:
                circuit_splits[row["circuit_id"]].add(row["split"])
            self.assertEqual(len(circuit_splits), 100)
            self.assertTrue(all(len(v) == 1 for v in circuit_splits.values()))
            for stream in ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed"):
                values = {r[stream] for r in rows}
                self.assertEqual(len(values), 100 if stream == "connectivity_seed" else 10000)
                self.assertTrue(all(not values & other for other in all_streams))
                self.assertFalse(values & {41, 42, 43})
                all_streams.append(values)

    def test_plan_and_source_download_cannot_release_production(self):
        config = campaign_config(CONFIG)
        # This negative fixture is independent of other circuits' live freeze state.
        for name in ("parameter_domain", "task_protocol", "observation_map"):
            config["scenarios"]["v1_visual"][name] = dict(frozen=False)
        self.assertIn("source_qualification", production_prerequisites(config, "v1_visual", None))
        fake_source = dict(status="PASS", scenario_id="v1_visual", coupled_core_neurons=230000,
                           source_revision="downloaded", evidence=["code"], checks={})
        missing = production_prerequisites(config, "v1_visual", fake_source)
        self.assertIn("numerical_integrity", missing)
        self.assertIn("parameter_domain_not_frozen", missing)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "identities.csv"
            source = Path(folder) / "source.json"
            source.write_text(json.dumps(fake_source))
            with self.assertRaisesRegex(ValueError, "NOT_QUALIFIED"):
                freeze_identities(config, "v1_visual", source, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
