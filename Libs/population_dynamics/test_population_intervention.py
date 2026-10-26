"""Contract checks for full-unit native pulse candidate ranking."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from . import population_intervention as intervention
from .mesocircuit_continuation import prepare_pulse_protocol
from .population_state import ControlledPopulationState
from .trajectory_scores import energy_score


def population_model():
    model = ControlledPopulationState(
        unit_ids=[13, 2, 41, 7, 19, 3, 29],
        unit_field_index=[0, 1, 0, 1, 0, 1, 0],
        field_names=["source_E", "source_I"],
        minimum_interval_ticks=[3, 4, 5, 3, 4, 5, 3],
        tick_seconds=.001, bin_seconds=.005, input_size=1,
        latent_channels=2, encoder_size=5, representation="recurrent",
        initialization_seed=41, unit_chunk_size=3,
    ).double().eval()
    with torch.no_grad():
        model.log_hazard_bias.fill_(5.)
    return model


class NativePulseEpisode:
    """Native observation stand-in with measured TC-arrival count bins."""

    def __init__(self, model, initial_emission_ms=(5., 17., 18., 20., 60.)):
        self.unit_ids = np.asarray(model.unit_ids)
        self.field_ids = np.asarray(model.field_names)
        self.unit_field_index = model.unit_field_index.cpu().numpy().copy()
        self.minimum_interval_ticks = model.minimum_interval_ticks.cpu().numpy().copy()
        self.dt_ms, self.bin_ms, self.bin_ticks, self.duration_ticks = 1., 5., 5, 80
        self.initial_emission_ms = initial_emission_ms
        self.pulse_ticks = np.asarray(initial_emission_ms, dtype=np.int64) + 3
        self.counts = np.zeros((16, len(self.unit_ids)), dtype=np.int64)
        self.counts[1::3, ::2] = 1
        self.stimulus = np.bincount(self.pulse_ticks // self.bin_ticks, minlength=16)[:, None]
        self.window_calls = []

    def window(self, start, stop):
        self.window_calls.append((start, stop))
        if start != 0 or stop % self.bin_ticks:
            raise ValueError("aligned native prefix required")
        return self.counts[:stop // self.bin_ticks].copy(), self.stimulus[:stop // self.bin_ticks].copy()

    def ready_at(self, tick):
        ready = np.zeros(len(self.unit_ids), dtype=np.int64)
        for unit in range(len(self.unit_ids)):
            emitted = np.flatnonzero(self.counts[:tick // self.bin_ticks, unit])
            if len(emitted):
                last = (int(emitted[-1]) + 1) * self.bin_ticks - 1
                ready[unit] = max(0, last + int(self.minimum_interval_ticks[unit]) - tick)
        return ready


class PopulationInterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = population_model()
        self.native = NativePulseEpisode(self.model)
        self.net = dict(thalamic_input=True, thalamic_input_type="pulses", th_delay_pulse_generator=3.,
                        th_pulse_start=8., th_interval=10.)
        self.sim = dict(sim_resolution=1., t_presim=0., t_sim=80.)
        self.initial = self.native.initial_emission_ms
        counts, controls = self.native.window(0, 20)
        self.history = dict(counts=counts[None], stimulus=controls[None],
                            ready_in_ticks=self.native.ready_at(20)[None])
        self.target = intervention.ReferenceTarget(
            field_rate_hz=np.stack((np.zeros((8, 2)), np.full((8, 2), 80.), np.full((8, 2), 200.))),
            field_names=self.model.field_names, scale_hz=np.array([10., 20.]),
            independent_block_ids=("source_14", "source_15", "source_16"), bin_seconds=.005,
        )
        self.candidates = (
            intervention.PulseCandidate("omit", ()),
            intervention.PulseCandidate("single", (30.,)),
            intervention.PulseCandidate("pair", (30., 40.)),
        )
        self.seeds, self.draws = (101, 102, 103), 3

    def design(self, **changes):
        arguments = dict(native=self.native, net=self.net, sim=self.sim,
            initial_emission_ms=self.initial, decision_time_ms=20., candidates=self.candidates,
            stream_seeds=self.seeds, draws=self.draws, pulse_cost_weight=.25)
        arguments.update(changes)
        return intervention.design_intervention(self.model, self.history, self.target, **arguments)

    def protocol(self, future=()):
        return prepare_pulse_protocol(self.net, self.sim, dict(initial_emission_ms=list(self.initial),
            decision_time_ms=20., future_emission_ms=list(future)))

    def test_streamed_full_unit_projection_matches_original_simulator_exactly(self):
        state = self.model.infer_state(self.history)
        controls = intervention.future_controls(self.protocol((30., 40.)), self.model, 8)
        before = torch.get_rng_state().clone()
        sampled = intervention.sample_candidate(self.model, state, controls, self.seeds, self.draws)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        streams = intervention.LatentRandomStreams.from_seeds(*self.seeds)
        steps = list(self.model.iter_simulation(state, controls, streams, self.draws, record_events=False))
        complete = torch.stack([step.counts[:, 0] for step in steps], dim=1)
        expected = torch.zeros((self.draws, 8, 2), dtype=torch.int64)
        expected.index_add_(-1, self.model.unit_field_index, complete)
        np.testing.assert_array_equal(sampled["field_counts"], expected.numpy())
        np.testing.assert_array_equal(sampled["unit_total_counts"], complete.sum(dim=1).numpy())
        np.testing.assert_array_equal(sampled["terminal_ready_in_ticks"], steps[-1].ready_in_ticks[:, 0].numpy())
        np.testing.assert_array_equal(sampled["field_rate_hz"], expected.numpy() / (np.array([4, 3]) * .005))
        self.assertEqual(sampled["generated_unit_count_values"], self.draws * 8 * 7)
        self.assertEqual(sampled["generated_bins"], 8)
        self.assertTrue(all(len(step.events) == 0 for step in steps))

    def test_reference_energy_averages_all_blocks_instead_of_target_mean(self):
        draws = np.stack((np.full((8, 2), 60.), np.full((8, 2), 100.), np.full((8, 2), 140.)))
        before = self.target.scale_hz.copy()
        score, per_block = intervention.reference_energy_score(draws, self.target, self.model)
        expected = [energy_score(draws / before, target / before) for target in self.target.field_rate_hz]
        np.testing.assert_allclose(per_block, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(score, float(np.mean(expected)), places=12)
        mean_score = energy_score(draws / before, self.target.field_rate_hz.mean(axis=0) / before)
        self.assertGreater(abs(score - mean_score), .1)
        permuted = replace(self.target, field_rate_hz=self.target.field_rate_hz[::-1],
                           independent_block_ids=self.target.independent_block_ids[::-1])
        other, other_rows = intervention.reference_energy_score(draws, permuted, self.model)
        self.assertAlmostEqual(score, other, places=12)
        np.testing.assert_allclose(other_rows, per_block[::-1], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(before, self.target.scale_hz)

    def test_native_emissions_and_delay_must_match_measured_tc_ticks(self):
        protocol = intervention.validate_native_controls(self.native, self.net, self.sim, self.initial)
        np.testing.assert_array_equal(np.asarray(protocol.initial_ticks) + protocol.delay_ticks, self.native.pulse_ticks)
        self.native.pulse_ticks = self.native.pulse_ticks.copy()
        self.native.pulse_ticks[0] += 1
        with self.assertRaisesRegex(ValueError, "actual source TC controls"):
            intervention.validate_native_controls(self.native, self.net, self.sim, self.initial)

    def test_committed_boundary_and_inflight_pulses_survive_omission(self):
        protocol = self.protocol()
        self.assertEqual(protocol.committed_ticks, (5, 17, 18, 20))
        self.assertEqual(protocol.effective_ticks, (5, 17, 18, 20))
        self.assertEqual(protocol.record()["inflight_tc_arrival_ms"], [21., 23.])
        controls = intervention.future_controls(protocol, self.model, 8)
        np.testing.assert_array_equal(controls[0, :, 0], [3., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(float(controls.sum()), 3.)

    def test_future_controls_use_half_open_endpoint_and_strict_future_emission(self):
        # Arrival60 lies exactly at the eight-bin endpoint and is excluded;
        # arrival59 belongs to its last half-open bin.
        controls = intervention.future_controls(self.protocol((56., 57.)), self.model, 8)
        np.testing.assert_array_equal(controls[0, :, 0], [3., 0., 0., 0., 0., 0., 0., 1.])
        with self.assertRaisesRegex(ValueError, "STRICTLY_FUTURE"):
            self.protocol((20.,))
        with self.assertRaisesRegex(ValueError, "scored response endpoint"):
            self.design(candidates=(intervention.PulseCandidate("endpoint", (57.,)),))

    def test_distinct_schedules_with_identical_subbin_controls_are_rejected(self):
        candidates = (intervention.PulseCandidate("early", (30.,)), intervention.PulseCandidate("late", (31.,)))
        with self.assertRaisesRegex(ValueError, "indistinguishable"):
            self.design(candidates=candidates)
        for candidates in (
            (intervention.PulseCandidate("same_a", (30.,)), intervention.PulseCandidate("same_b", (30.,))),
            (intervention.PulseCandidate("duplicate", ()), intervention.PulseCandidate("duplicate", (30.,))),
            (intervention.PulseCandidate("list", [30.]),),
        ):
            with self.subTest(candidates=candidates):
                with self.assertRaises(ValueError):
                    self.design(candidates=candidates)

    def test_real_design_scores_every_candidate_preserves_rng_and_full_terminal_state(self):
        rng_before = torch.get_rng_state().clone()
        report, predictions = self.design()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual([row["candidate"] for row in report["candidates"]], [c.name for c in self.candidates])
        self.assertEqual(set(predictions), {c.name for c in self.candidates})
        self.assertEqual(report["reference_blocks"], list(self.target.independent_block_ids))
        self.assertTrue(report["all_candidate_outcomes_retained"])
        self.assertTrue(report["full_unit_outputs_generated"])
        self.assertFalse(report["native_future_noise_pairing_verified"])
        self.assertFalse(report["success_tolerance_qualified"])
        self.assertFalse(report["inverse_intervention_qualified"])
        self.assertFalse(report["native_execution_performed"])
        for row in report["candidates"]:
            prediction = predictions[row["candidate"]]
            self.assertEqual(prediction["terminal_ready_in_ticks"].shape, (self.draws, 7))
            self.assertTrue(np.all(prediction["terminal_ready_in_ticks"] < self.native.minimum_interval_ticks))
            self.assertEqual(len(row["reference_block_scores"]), 3)
            self.assertEqual(row["generated_unit_count_values"], self.draws * 8 * 7)
            self.assertAlmostEqual(row["objective"], row["reference_energy_score"] + .25 * row["future_pulse_count"])
        selected = min(report["candidates"], key=lambda r: (r["objective"], r["future_pulse_count"], r["candidate"]))
        self.assertEqual(report["selected_candidate"], selected["candidate"])

    def test_design_rejects_changed_counts_ready_and_tc_prefix(self):
        for name in ("counts", "ready_in_ticks", "stimulus"):
            with self.subTest(history=name):
                original = self.history
                self.history = {key: value.copy() for key, value in original.items()}
                self.history[name].flat[0] += 1
                try:
                    with self.assertRaises(ValueError):
                        self.design()
                finally:
                    self.history = original

    def test_design_does_not_read_future_cortical_counts(self):
        before, predictions = self.design()
        self.native.counts[4:] += 3
        after, changed = self.design()
        self.assertEqual(before["selected_candidate"], after["selected_candidate"])
        np.testing.assert_array_equal(
            [r["objective"] for r in before["candidates"]], [r["objective"] for r in after["candidates"]])
        for candidate in predictions:
            np.testing.assert_array_equal(predictions[candidate]["field_counts"], changed[candidate]["field_counts"])

    def test_malformed_reference_source_scale_and_independent_blocks_are_rejected(self):
        for target in (
            replace(self.target, field_names=self.target.field_names[::-1]),
            replace(self.target, scale_hz=np.array([0., 20.])),
            replace(self.target, scale_hz=np.array([np.nan, 20.])),
            replace(self.target, independent_block_ids=("source_14", "source_14", "source_16")),
            replace(self.target, independent_block_ids=("source_14", "source_15")),
            replace(self.target, bin_seconds=.01),
        ):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    intervention.target_arrays(target, self.model)
        original = self.native.unit_ids
        self.native.unit_ids = original[::-1]
        with self.assertRaisesRegex(ValueError, "observation map"):
            self.design()
        self.native.unit_ids = original
        original_map = self.native.unit_field_index
        self.native.unit_field_index = 1 - original_map
        with self.assertRaisesRegex(ValueError, "observation map"):
            self.design()

    def test_empty_field_and_invalid_sampling_streams_are_rejected(self):
        state = self.model.infer_state(self.history)
        controls = intervention.future_controls(self.protocol(), self.model, 8)
        for seeds, draws in (((1, 1, 2), 3), ((1, 2, 3), 1), ((1, 2), 3)):
            with self.subTest(seeds=seeds, draws=draws):
                with self.assertRaises(ValueError):
                    intervention.sample_candidate(self.model, state, controls, seeds, draws)
        empty = ControlledPopulationState(unit_ids=[1, 2], unit_field_index=[0, 0],
            field_names=["source_E", "empty"], minimum_interval_ticks=[3, 3],
            tick_seconds=.001, bin_seconds=.005, input_size=1, latent_channels=2,
            encoder_size=5, initialization_seed=41).double()
        boundary = empty.infer_state(dict(counts=np.zeros((1, 4, 2), dtype=np.int64),
            stimulus=np.zeros((1, 4, 1)), ready_in_ticks=np.zeros((1, 2), dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "unit denominator"):
            intervention.sample_candidate(empty, boundary, np.zeros((1, 8, 1)), self.seeds, 3)


if __name__ == "__main__":
    unittest.main()
