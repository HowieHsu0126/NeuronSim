"""Count/latent interfaces and checkpoint replay on a bounded S4 fixture."""
import io
import unittest

import torch

from .latent_energy import ControlledLatentEnergy


class LatentEnergyTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(41)
        return ControlledLatentEnergy((10, 20, 30, 40), complete_episode_bins=24,
            maximum_prefix_bins=18, forecast_bins=6, latent_channels=4, hidden_channels=16,
            encoder_blocks=1, decoder_blocks=2, linear_layers_per_block=4,
            generator_options=dict(encoder_width=8, encoder_depth=1, encoder_heads=2,
                decoder_width=8, decoder_depth=1, decoder_heads=2, score_mlp_width=12,
                score_mlp_depth=1, noise_channels=4))

    def counts(self):
        return (torch.arange(2 * 24 * 4).reshape(2, 24, 4) % 3).float()

    def test_decoder_trains_without_late_s4_parameter_reconfiguration(self):
        model = self.model()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        counts = self.counts()
        mask = (torch.arange(counts.numel()).reshape(counts.transpose(1, 2).shape) % 2).bool()
        terms = model.autoencoder_loss(counts, mask=mask)
        terms['loss'].backward()
        self.assertTrue(torch.isfinite(terms['loss']))
        for block in (*model.autoencoder.ae_net.encoder, *model.autoencoder.ae_net.decoder):
            self.assertTrue(block.bidirectional)
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in block.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters()))
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_prefix_is_separately_encoded_and_target_transform_is_registered(self):
        model = self.model().eval()
        counts = self.counts()
        controls = torch.linspace(0, 1, 48).reshape(2, 24, 1)
        with self.assertRaisesRegex(ValueError, 'scale'):
            model.infer_state(counts[:, :12], controls[:, :12])
        model.set_latent_scale(torch.zeros(1, 4, 1), torch.ones(1, 4, 1))
        before = model.infer_state(counts[:, :12], controls[:, :12])
        counts[:, 12:] += 20
        controls[:, 12:] += 10
        repeated = model.infer_state(counts[:, :12], controls[:, :12])
        self.assertTrue(torch.equal(before.latent, repeated.latent))
        self.assertTrue(torch.equal(before.controls, repeated.controls))
        self.assertEqual(before.latent.shape, (2, 12, 4))
        self.assertEqual(before.unit_ids, (10, 20, 30, 40))
        target = model.encode_target(counts[:, 12:18])
        self.assertEqual(target.shape, (2, 6, 4))
        with self.assertRaisesRegex(ValueError, 'forecast'):
            model.encode_target(counts[:, 12:17])

    def test_complete_unit_sampling_and_saved_model_replay(self):
        model = self.model().eval()
        model.set_latent_scale(torch.zeros(1, 4, 1), torch.ones(1, 4, 1))
        counts = self.counts()
        state = model.infer_state(counts[:, :12], torch.zeros(2, 12, 1))
        stimulus = torch.ones(2, 6, 1)
        rng_before = torch.random.get_rng_state().clone()

        def simulate(instance, condition):
            return instance.simulate(condition, stimulus, steps=3,
                latent_rng=torch.Generator().manual_seed(42),
                observation_rng=torch.Generator().manual_seed(43))

        result = simulate(model, state)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertEqual(result['counts'].shape, (2, 6, 4))
        self.assertEqual(result['counts'].dtype, torch.int64)
        self.assertTrue(torch.all(result['counts'] >= 0))
        self.assertEqual(result['expected_counts'].shape, result['counts'].shape)
        self.assertEqual(result['bin_seconds'], .005)
        self.assertFalse(result['native_refractory_support_qualified'])
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        repeated_state = restored.infer_state(counts[:, :12], torch.zeros(2, 12, 1))
        repeated = simulate(restored, repeated_state)
        for name in ('counts', 'expected_counts', 'latent'):
            self.assertTrue(torch.equal(result[name], repeated[name]), name)


if __name__ == '__main__':
    unittest.main()
