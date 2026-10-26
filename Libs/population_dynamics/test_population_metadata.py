"""Shared parameter contracts across distinct source circuit maps."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from .population_state import ControlledPopulationState
from .poisson_state_space import LatentRandomStreams


def descriptors(units=7, fields=3):
    schema = [dict(name="source_fraction", unit="dimensionless",
                   meaning="fixture source descriptor", transform="identity")]
    return dict(unit_feature_schema=schema, field_feature_schema=copy.deepcopy(schema),
                source_identity=dict(circuit_id=f"fixture_{units}_{fields}", source_revision="fixture_source",
                                     connectivity_identity="fixture_connectivity", parameter_identity="fixture_parameter"),
                control_features=[dict(name="current", unit="nA", role="known_current")],
                unit_features=[[i / units] for i in range(units)],
                field_features=[[i / fields] for i in range(fields)])


def model(units=7, fields=3):
    return ControlledPopulationState(unit_ids=list(range(units)),
        unit_field_index=[i % fields for i in range(units)],
        field_names=[f"source_{i}" for i in range(fields)], minimum_interval_ticks=[3] * units,
        tick_seconds=.001, bin_seconds=.005, input_size=1, latent_channels=2, encoder_size=5,
        initialization_seed=41, unit_chunk_size=3, metadata=descriptors(units, fields)).double()


class PopulationMetadataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = model()
        self.counts = torch.zeros(1, 8, 7, dtype=torch.float64)
        self.counts[:, 1::3, ::2] = 1
        self.control = torch.zeros(1, 8, 1, dtype=torch.float64)

    def test_all_fitted_parameters_are_map_independent_and_rng_is_preserved(self):
        before = torch.get_rng_state().clone()
        other = model(11, 4)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual({n: tuple(p.shape) for n, p in self.model.named_parameters()},
                         {n: tuple(p.shape) for n, p in other.named_parameters()})
        forbidden = {"unit_loading", "log_hazard_bias", "field_mix_logits", "control_weight",
                     "transition_bias", "decay_logit", "process_log_variance",
                     "initial_mean", "initial_log_variance"}
        self.assertFalse(forbidden & set(dict(self.model.named_parameters())))

    def test_shared_likelihood_updates_rebind_to_new_identity_and_shape(self):
        old = self.model.shared_unit_readout.weight.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        terms = self.model.sequence_loss(self.counts, self.control, torch.full_like(self.counts, 5),
                                         torch.Generator().manual_seed(987))
        terms["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(old, self.model.shared_unit_readout.weight))
        other = self.model.for_circuit(unit_ids=list(range(100, 111)), unit_field_index=[i % 4 for i in range(11)],
            field_names=[f"other_{i}" for i in range(4)], minimum_interval_ticks=[4] * 11,
            metadata=descriptors(11, 4))
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, dict(other.named_parameters())[name]))
        self.assertEqual(other.unit_ids, tuple(range(100, 111)))
        self.assertEqual(other.field_unit_counts.tolist(), [3, 3, 3, 2])
        self.assertEqual(other.minimum_interval_ticks.tolist(), [4] * 11)
        terms = other.sequence_loss(torch.zeros(1, 8, 11, dtype=torch.float64), self.control,
                                   torch.full((1, 8, 11), 5), torch.Generator().manual_seed(987))
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))

    def test_training_views_share_optimizer_parameters_across_different_maps(self):
        before_rng = torch.get_rng_state().clone()
        other = self.model.training_for_circuit(unit_ids=list(range(100, 111)),
            unit_field_index=[i % 4 for i in range(11)], field_names=[f"other_{i}" for i in range(4)],
            minimum_interval_ticks=[4] * 11, metadata=descriptors(11, 4))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        for name, value in self.model.named_parameters():
            self.assertIs(value, dict(other.named_parameters())[name])
        for name, value in self.model.named_buffers():
            if name in dict(other.named_buffers()):
                self.assertIsNot(value, dict(other.named_buffers())[name])
        copied = self.model.for_circuit(unit_ids=list(other.unit_ids),
            unit_field_index=other.unit_field_index.tolist(), field_names=list(other.field_names),
            minimum_interval_ticks=other.minimum_interval_ticks.tolist(), metadata=other.metadata)
        self.assertIsNot(copied.shared_unit_readout.weight, self.model.shared_unit_readout.weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        for current, counts in ((other, torch.zeros(1, 8, 11, dtype=torch.float64)),
                                (self.model, self.counts)):
            old = self.model.shared_unit_readout.weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            current.sequence_loss(counts, self.control, torch.full_like(counts, 5),
                                  torch.Generator().manual_seed(987))["loss"].backward()
            optimizer.step()
            self.assertFalse(torch.equal(old, self.model.shared_unit_readout.weight))
        self.assertEqual(int(optimizer.state[self.model.shared_unit_readout.weight]["step"]), 2)
        other.eval()
        self.assertTrue(self.model.training)
        boundary = self.model.infer_state(dict(counts=self.counts, stimulus=self.control,
            ready_in_ticks=torch.zeros(1, 7, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "boundary"):
            next(other.iter_simulation(boundary, self.control[:, :1],
                LatentRandomStreams.from_seeds(11, 12, 13), 1))

    def test_prefix_isolation_and_source_permutation_equivariance(self):
        mean, variance = self.model.encode(self.counts, self.control)
        changed = self.counts.clone(); changed[:, 5:] += 2
        changed_u = self.control.clone(); changed_u[:, 5:] += 3
        after, after_v = self.model.encode(changed, changed_u)
        torch.testing.assert_close(mean[:, :5], after[:, :5], rtol=0, atol=0)
        torch.testing.assert_close(variance[:, :5], after_v[:, :5], rtol=0, atol=0)
        unit_order = torch.tensor([6, 4, 2, 0, 5, 3, 1]); field_order = torch.tensor([2, 0, 1])
        inverse_field = torch.argsort(field_order)
        metadata = copy.deepcopy(self.model.metadata)
        metadata["unit_features"] = [metadata["unit_features"][i] for i in unit_order]
        metadata["field_features"] = [metadata["field_features"][i] for i in field_order]
        other = self.model.for_circuit(unit_ids=[self.model.unit_ids[i] for i in unit_order],
            unit_field_index=inverse_field[self.model.unit_field_index[unit_order]].tolist(),
            field_names=[self.model.field_names[i] for i in field_order],
            minimum_interval_ticks=[3] * 7, metadata=metadata)
        permuted, permuted_v = other.encode(self.counts[:, :, unit_order], self.control)
        torch.testing.assert_close(permuted, mean[:, :, field_order], rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(permuted_v, variance[:, :, field_order], rtol=1e-14, atol=1e-14)
        z = torch.randn(1, 3, 2, dtype=torch.float64)
        transition = self.model.transition_mean(z, self.control[:, 0])
        torch.testing.assert_close(other.transition_mean(z[:, field_order], self.control[:, 0]),
                                   transition[:, field_order], rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(other.log_hazard(z[:, field_order]), self.model.log_hazard(z)[:, unit_order],
                                   rtol=0, atol=0)

    def test_checkpoint_replay_and_native_support(self):
        with torch.no_grad(): self.model.shared_unit_readout.bias[-1] = 6
        boundary = self.model.infer_state(dict(counts=self.counts, stimulus=self.control,
                                               ready_in_ticks=torch.zeros(1, 7, dtype=torch.long)))
        future = torch.zeros(1, 5, 1, dtype=torch.float64)
        def sample(value):
            return list(value.iter_simulation(boundary, future, LatentRandomStreams.from_seeds(11, 12, 13), 2))
        original = sample(self.model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            self.model.save(path)
            loaded = ControlledPopulationState.load(path)
            for first, second in zip(original, sample(loaded)):
                self.assertTrue(torch.equal(first.counts, second.counts))
                self.assertTrue(torch.equal(first.events, second.events))
                self.assertTrue(torch.equal(first.latent, second.latent))
            payload = torch.load(path, weights_only=True)
            payload["state_dict"]["unit_field_index"][0] = 1
            changed = Path(folder) / "changed.pt"
            torch.save(payload, changed)
            with self.assertRaisesRegex(ValueError, "source buffers changed"):
                ControlledPopulationState.load(changed)
        events = torch.cat([s.events for s in original])
        for draw in range(2):
            for unit in range(7):
                ticks = events[(events[:, 0] == draw) & (events[:, 2] == unit), 3].sort().values
                self.assertTrue(torch.all(torch.diff(ticks) >= 3))

    def test_changed_measurement_contract_rejects_before_parameter_transfer(self):
        for key in ("unit_feature_schema", "field_feature_schema", "control_features"):
            metadata = descriptors()
            metadata[key][0]["unit"] = "different_unit"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "measurement contract changed"):
                self.model.for_circuit(unit_ids=list(range(7)), unit_field_index=[i % 3 for i in range(7)],
                    field_names=list(self.model.field_names), minimum_interval_ticks=[3] * 7, metadata=metadata)
        config = self.model.configuration(); config.pop("metadata")
        with self.assertRaisesRegex(ValueError, "cannot transfer"):
            ControlledPopulationState(**config).for_circuit(unit_ids=list(range(7)),
                unit_field_index=[i % 3 for i in range(7)], field_names=list(self.model.field_names),
                minimum_interval_ticks=[3] * 7, metadata=descriptors())
        metadata = descriptors(); metadata["unit_features"][0][0] = 9
        other = self.model.for_circuit(unit_ids=list(range(7)), unit_field_index=[i % 3 for i in range(7)],
            field_names=list(self.model.field_names), minimum_interval_ticks=[3] * 7, metadata=metadata)
        boundary = self.model.infer_state(dict(counts=self.counts, stimulus=self.control,
                                               ready_in_ticks=torch.zeros(1, 7, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "observation map mismatch"):
            next(other.iter_simulation(boundary, self.control[:, :1],
                                       LatentRandomStreams.from_seeds(11, 12, 13), 1))
        metadata = descriptors(); metadata["source_identity"]["connectivity_identity"] = "another_realization"
        other = self.model.for_circuit(unit_ids=list(range(7)), unit_field_index=[i % 3 for i in range(7)],
            field_names=list(self.model.field_names), minimum_interval_ticks=[3] * 7, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "observation map mismatch"):
            next(other.iter_simulation(boundary, self.control[:, :1],
                                       LatentRandomStreams.from_seeds(11, 12, 13), 1))

    def test_invalid_static_descriptors_reject(self):
        for change in ("nonfinite", "shape", "unlabeled", "outcome"):
            metadata = descriptors()
            if change == "nonfinite": metadata["unit_features"][0][0] = float("nan")
            if change == "shape": metadata["field_features"].pop()
            if change == "unlabeled": metadata["unit_feature_schema"][0].pop("meaning")
            if change == "outcome": metadata["task_outcome"] = "success"
            config = self.model.configuration(); config["metadata"] = metadata
            with self.subTest(change=change), self.assertRaises(ValueError):
                ControlledPopulationState(**config)

    def test_no_control_metadata_mode_runs_full_likelihood(self):
        config = self.model.configuration()
        config["input_size"] = 0
        config["metadata"]["control_features"] = []
        self.assertEqual(len(self.model.metadata["control_features"]), 1)
        other = ControlledPopulationState(**config).double()
        terms = other.sequence_loss(self.counts, torch.empty(1, 8, 0, dtype=torch.float64),
                                   torch.full_like(self.counts, 5), torch.Generator().manual_seed(987))
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))


if __name__ == "__main__":
    unittest.main()
