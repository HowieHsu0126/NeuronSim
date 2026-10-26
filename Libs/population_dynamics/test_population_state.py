import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from .population_state import ControlledPopulationState, PopulationBoundary
from .poisson_state_space import LatentRandomStreams
from .closure_selection import assess_state, select_development_state


def example(representation="recurrent", units=7):
    return ControlledPopulationState(unit_ids=list(range(units)),
        unit_field_index=[i % 2 for i in range(units)], field_names=["source_E", "source_I", "empty"],
        minimum_interval_ticks=[3] * units, tick_seconds=.001, bin_seconds=.005,
        input_size=1, latent_channels=2, encoder_size=5, representation=representation,
        initialization_seed=41, unit_chunk_size=3)


class PopulationModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = example().double()
        self.y = torch.zeros(2, 8, 7, dtype=torch.float64)
        self.y[:, 1::3, ::2] = 1
        self.u = torch.linspace(0, 1, 16, dtype=torch.float64).reshape(2, 8, 1)

    def test_constructor_does_not_consume_global_rng(self):
        before = torch.get_rng_state().clone()
        example()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_causal_prefix_and_explicit_sizes(self):
        mean, variance = self.model.encode(self.y, self.u)
        changed = self.y.clone(); changed[:, 5:] += 3
        changed_u = self.u.clone(); changed_u[:, 5:] += 9
        after, after_v = self.model.encode(changed, changed_u)
        torch.testing.assert_close(mean[:, :5], after[:, :5], rtol=0, atol=0)
        torch.testing.assert_close(variance[:, :5], after_v[:, :5], rtol=0, atol=0)
        features = self.model.field_features(self.y, self.u)
        self.assertTrue(torch.equal(features[:, :, 2, :3], torch.zeros_like(features[:, :, 2, :3])))
        self.assertEqual(self.model.field_unit_counts.tolist(), [4, 3, 0])

    def test_native_likelihood_updates_and_stream_state(self):
        generator = torch.Generator().manual_seed(987)
        before = generator.get_state().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        old = self.model.unit_loading.detach().clone()
        terms = self.model.sequence_loss(self.y, self.u, torch.full_like(self.y, 5), generator)
        terms["loss"].backward(); optimizer.step()
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))
        self.assertFalse(torch.equal(old, self.model.unit_loading))
        self.assertFalse(torch.equal(before, generator.get_state()))

    def test_field_interaction_and_control_response(self):
        z = torch.zeros(1, 3, 2, dtype=torch.float64)
        zero = self.model.transition_mean(z, torch.zeros(1, 1, dtype=torch.float64))
        z[:, 0] = 1
        mixed = self.model.transition_mean(z, torch.zeros(1, 1, dtype=torch.float64))
        self.assertTrue(torch.any(mixed[:, 1] != zero[:, 1]))
        controlled = self.model.transition_mean(z, torch.ones(1, 1, dtype=torch.float64))
        self.assertTrue(torch.any(controlled != mixed))

    def test_joint_samples_replay_serialization_and_support(self):
        with torch.no_grad(): self.model.log_hazard_bias.fill_(5.)
        boundary = self.model.infer_state(dict(counts=self.y, stimulus=self.u,
                                               ready_in_ticks=torch.zeros(2, 7, dtype=torch.long)))
        future = torch.zeros(2, 5, 1, dtype=torch.float64)
        def generate(model, state=boundary):
            return list(model.iter_simulation(state, future, LatentRandomStreams.from_seeds(11, 12, 13), 3))
        a, b = generate(self.model), generate(self.model)
        self.assertEqual(len(a), 5)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x.counts, y.counts))
            self.assertTrue(torch.equal(x.events, y.events))
            self.assertEqual(x.counts.shape, (3, 2, 7))
        events = torch.cat([s.events for s in a]).numpy()
        for draw in range(3):
            for batch in range(2):
                for unit in range(7):
                    ticks = np.sort(events[(events[:, 0] == draw) & (events[:, 1] == batch) & (events[:, 2] == unit), 3])
                    self.assertTrue(np.all(np.diff(ticks) >= 3))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            self.model.save(path)
            loaded = ControlledPopulationState.load(path)
            for x, y in zip(a, generate(loaded)):
                self.assertTrue(torch.equal(x.counts, y.counts))
                self.assertTrue(torch.equal(x.latent, y.latent))
            with self.assertRaises(FileExistsError): self.model.save(path)

    def test_fixed_encoder_does_not_read_old_bins(self):
        model = example("fixed").double()
        mean, _ = model.encode(self.y, self.u)
        changed = self.y.clone(); changed[:, :4] += 4
        after, _ = model.encode(changed, self.u)
        torch.testing.assert_close(mean[:, 4:], after[:, 4:], rtol=0, atol=0)

    def test_rejects_invalid_support_and_metadata(self):
        with self.assertRaises(ValueError):
            self.model.sequence_loss(self.y, self.u, torch.zeros_like(self.y), torch.Generator())
        with self.assertRaises(ValueError):
            self.model.infer_state(dict(counts=self.y, stimulus=self.u,
                                        ready_in_ticks=torch.full((2, 7), 3, dtype=torch.long)))
        with self.assertRaises(ValueError):
            self.model.infer_state(dict(counts=self.y, stimulus=self.u, ready_in_ticks=torch.zeros(2, 7, dtype=torch.long)),
                                   dict(unit_ids=list(range(7)), field_names=["wrong"]))


class ClosureWitnessTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(187)
        state = rng.normal(size=(400, 1)); history = rng.normal(size=(400, 1))
        controls = rng.integers(0, 2, size=(400, 1)).astype(float)
        targets = state + 3 * history * controls
        return dict(fit_state=state[:300], selection_state=state[300:],
                    fit_history=history[:300], selection_history=history[300:],
                    fit_controls=controls[:300], selection_controls=controls[300:],
                    fit_targets=targets[:300], selection_targets=targets[300:],
                    fit_blocks=np.repeat(np.arange(6), 50), selection_blocks=np.repeat([6, 7], 50),
                    penalty=.01, target_names=["paired_control_response"])

    def test_detects_control_dependent_omission_and_keeps_all_blocks(self):
        d = self.data(); result = assess_state(**d)
        self.assertGreater(result["mean_history_gain"][0], .5)
        self.assertEqual(result["selection_blocks"], 2)
        self.assertFalse(result["full_joint_distribution_certified"])
        decision = select_development_state({"small": result}, {"small": 1.}, {"paired_control_response": .01})
        self.assertIsNone(decision["selected_state"])

    def test_retained_history_removes_this_witness(self):
        d = self.data()
        d["fit_state"] = np.concatenate((d["fit_state"], d["fit_history"]), 1)
        d["selection_state"] = np.concatenate((d["selection_state"], d["selection_history"]), 1)
        result = assess_state(**d)
        self.assertLess(abs(result["mean_history_gain"][0]), 1e-6)
        decision = select_development_state({"retained": result}, {"retained": 2.}, {"paired_control_response": .01})
        self.assertEqual(decision["selected_state"], "retained")
        self.assertFalse(decision["scientific_closure_qualified"])

    def test_rejects_block_leakage_and_unfrozen_tolerances(self):
        d = self.data(); d["selection_blocks"][:] = 0
        with self.assertRaisesRegex(ValueError, "disjoint"): assess_state(**d)
        result = assess_state(**self.data())
        with self.assertRaisesRegex(ValueError, "predeclared"):
            select_development_state({"a": result}, {"a": 1.}, {"paired_control_response": None})


if __name__ == "__main__":
    unittest.main()
