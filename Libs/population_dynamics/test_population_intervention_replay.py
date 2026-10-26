"""Disposable real-model fixtures for saved-array pulse-search replay."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import population_intervention as search
from . import population_intervention_replay as replay
from .provenance import sha256_file
from .test_population_intervention import NativePulseEpisode, population_model


class PulseSearchReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name)
        self.model = population_model()
        self.native = NativePulseEpisode(self.model)
        self.net = dict(thalamic_input=True, thalamic_input_type="pulses", th_delay_pulse_generator=3.,
                        th_pulse_start=8., th_interval=10.)
        self.sim = dict(sim_resolution=1., t_presim=0., t_sim=80.)
        self.initial = self.native.initial_emission_ms
        counts, controls = self.native.window(0, 20)
        self.history = dict(counts=counts[None], stimulus=controls[None], ready_in_ticks=self.native.ready_at(20)[None])
        self.target = search.ReferenceTarget(
            np.stack((np.zeros((8, 2)), np.full((8, 2), 80.), np.full((8, 2), 200.))),
            self.model.field_names, np.array([10., 20.]), ("source_14", "source_15", "source_16"), .005)
        candidates = (search.PulseCandidate("omit", ()), search.PulseCandidate("single", (30.,)),
                      search.PulseCandidate("pair", (30., 40.)))
        self.report, self.predictions = search.design_intervention(self.model, self.history, self.target,
            native=self.native, net=self.net, sim=self.sim, initial_emission_ms=self.initial,
            decision_time_ms=20., candidates=candidates, stream_seeds=(101, 102, 103), draws=3, pulse_cost_weight=.25)
        self.reference = dict(field_rate_hz=self.target.field_rate_hz, scale_hz=self.target.scale_hz,
                              field_ids=np.asarray(self.target.field_names), block_ids=np.asarray(self.target.independent_block_ids))
        np.savez_compressed(self.results / "reference.npz", **self.reference)
        for name, prediction in self.predictions.items():
            arrays = {key: value for key, value in prediction.items() if isinstance(value, np.ndarray)}
            arrays.update(unit_ids=np.asarray(self.model.unit_ids), field_ids=np.asarray(self.model.field_names))
            np.savez_compressed(self.results / (name + ".npz"), **arrays)
        keys = ("candidate", "reference_energy_score", "future_pulse_count", "objective",
                "generation_and_projection_seconds", "generated_unit_count_values")
        with (self.results / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row[key] for key in keys} for row in self.report["candidates"])
        self.report["artifacts"] = {path.name: sha256_file(path) for path in self.results.iterdir()}

    def perform(self):
        return replay.replay_candidates(self.model, self.native, self.report, self.reference,
            net=self.net, sim=self.sim, initial_emission_ms=self.initial, decision_time_ms=20., results=self.results)

    def saved(self, name):
        return replay._arrays(self.results / (name + ".npz"))

    def test_direct_simulator_reproduces_every_saved_candidate_array(self):
        before = torch.get_rng_state().clone()
        evidence = self.perform()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(evidence["candidate_count"], 3)
        self.assertEqual(evidence["reference_blocks_compared"], 3)
        self.assertEqual(evidence["selected_candidate"], self.report["selected_candidate"])
        self.assertTrue(evidence["saved_field_counts_exact"])
        self.assertTrue(evidence["saved_field_rates_exact"])
        self.assertTrue(evidence["saved_all_unit_totals_exact"])
        self.assertTrue(evidence["saved_terminal_unit_clocks_exact"])
        self.assertFalse(evidence["per_bin_full_unit_counts_saved"])
        self.assertFalse(evidence["per_bin_full_unit_counts_exact_replay_claimed"])

    def test_replay_is_independent_of_search_sampler_and_ranker(self):
        with patch.object(search, "sample_candidate", side_effect=AssertionError("search sampler reused")), \
             patch.object(search, "design_intervention", side_effect=AssertionError("search ranker reused")):
            evidence = self.perform()
        self.assertEqual(evidence["candidate_count"], 3)

    def test_rejects_each_tampered_saved_array_and_extra_unsaved_counts(self):
        original = self.saved("omit")
        for key in original:
            with self.subTest(array=key):
                changed = {name: values.copy() for name, values in original.items()}
                if changed[key].dtype.kind in "US":
                    changed[key] = changed[key][::-1]
                else:
                    changed[key].flat[0] += 1
                np.savez_compressed(self.results / "omit.npz", **changed)
                with self.assertRaisesRegex(ValueError, "saved array replay differs"):
                    self.perform()
        changed = dict(original, per_bin_unit_counts=np.zeros((3, 8, 7), dtype=np.int64))
        np.savez_compressed(self.results / "omit.npz", **changed)
        with self.assertRaisesRegex(ValueError, "schema"):
            self.perform()

    def test_hash_verifier_rejects_changed_payload_and_missing_candidates(self):
        replay.verify_artifacts(self.results, self.report, [row["candidate"] for row in self.report["candidates"]])
        path = self.results / "pair.npz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            replay.verify_artifacts(self.results, self.report, ["omit", "single", "pair"])
        with self.assertRaisesRegex(ValueError, "declared pulse-search artifacts"):
            replay.verify_artifacts(self.results, self.report, ["omit", "single"])

    def test_rejects_changed_protocol_score_and_selected_candidate(self):
        for key, value in (("reference_energy_score", 1.e6), ("reference_block_scores", [0., 0., 0.]),
                           ("objective", 1.e6), ("generated_unit_count_values", 1)):
            with self.subTest(report=key):
                original = self.report["candidates"][0][key]
                self.report["candidates"][0][key] = value
                with self.assertRaisesRegex(ValueError, "saved reference scores"):
                    self.perform()
                self.report["candidates"][0][key] = original
        original = self.report["candidates"][0]["protocol"]["committed_emission_ms"]
        self.report["candidates"][0]["protocol"]["committed_emission_ms"] = []
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            self.perform()
        self.report["candidates"][0]["protocol"]["committed_emission_ms"] = original
        self.report["selected_candidate"] = "missing"
        with self.assertRaisesRegex(ValueError, "selected candidate"):
            self.perform()

    def test_rejects_changed_native_boundary_controls_and_unit_identity(self):
        self.native.stimulus[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "past TC bins"):
            self.perform()
        self.native.stimulus[0, 0] -= 1
        self.native.pulse_ticks = self.native.pulse_ticks + 1
        with self.assertRaisesRegex(ValueError, "original source TC"):
            self.perform()
        self.native.pulse_ticks = self.native.pulse_ticks - 1
        self.native.unit_ids = self.native.unit_ids[::-1]
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.perform()

    def test_rejects_reference_identity_scale_and_duplicate_blocks(self):
        for key, value in (("field_ids", np.asarray(["wrong", "source_I"])),
                           ("scale_hz", np.array([0., 20.])),
                           ("block_ids", np.array(["source_14", "source_14", "source_16"]))):
            with self.subTest(reference=key):
                original = self.reference[key]
                self.reference[key] = value
                with self.assertRaisesRegex(ValueError, "reference source"):
                    self.perform()
                self.reference[key] = original

    def test_candidate_csv_is_compared_with_machine_readable_report(self):
        path = self.results / "candidates.csv"
        text = path.read_text()
        path.write_text(text.replace("omit,", "unknown,", 1))
        with self.assertRaisesRegex(ValueError, "CSV identity"):
            self.perform()

    def test_changed_config_is_rejected_before_model_or_native_loading(self):
        config = dict(scope="fixture")
        config_path = self.results / "config.yaml"
        config_path.write_text("scope: fixture\n")
        report = dict(status="ENGINEERING_NATIVE_PULSE_SEARCH_EXECUTED", configuration=dict(scope="changed"),
                      training_performed=False, mechanistic_simulation_performed=False,
                      native_execution_performed=False, inverse_intervention_qualified=False, scientific_claim_qualified=False)
        (self.results / "analysis.json").write_text(json.dumps(report))
        receipt_path = self.results / "receipt.json"
        receipt_path.write_text("{}")
        with patch.object(replay, "validate_config", return_value=None), \
             patch.object(replay.ControlledPopulationState, "load", side_effect=AssertionError("model loaded")):
            with self.assertRaisesRegex(ValueError, "engineering-only pulse search"):
                replay.replay(self.results, self.results, self.results / "missing.pt", receipt_path,
                    self.results / "missing_graph.json", config_path, self.results, self.results / "replay.json")
        self.assertFalse((self.results / "replay.json").exists())


if __name__ == "__main__":
    unittest.main()
