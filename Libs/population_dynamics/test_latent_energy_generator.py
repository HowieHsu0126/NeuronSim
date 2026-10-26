"""Small diagnostic fixtures for energy loss, conditioning and replay contracts."""
import inspect
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from .latent_energy_generator import (
    LatentEnergyGenerator, _AdaLNBlock, _EnergyHead, cosine_commit_counts, energy_score_terms,
)


class LatentEnergyGeneratorTests(unittest.TestCase):
    def activate(self, module):
        # Nonzero diagnostic parameters distinguish conditioning paths without
        # changing the production source-style zero initialization.
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if 'modulation' in name or name.startswith('output.'):
                    nn.init.normal_(parameter, std=0.1)

    def model(self, *, nonzero=True, **overrides):
        torch.manual_seed(41)
        options = dict(encoder_width=8, encoder_depth=1, encoder_heads=2,
                       decoder_width=8, decoder_depth=1, decoder_heads=2, mlp_ratio=2,
                       score_mlp_width=12, score_mlp_depth=1, noise_channels=4)
        options.update(overrides)
        model = LatentEnergyGenerator(2, 1, 16, **options)
        if nonzero:
            self.activate(model.energy_head)
        return model

    def inputs(self, past=3, future=5):
        prefix = torch.linspace(-1, 1, 2 * past * 2).reshape(2, past, 2)
        controls = torch.linspace(0, 1, 2 * (past + future)).reshape(2, past + future, 1)
        target = torch.linspace(-0.5, 0.8, 2 * future * 2).reshape(2, future, 2)
        return prefix, controls, target

    def rng(self, seed=42):
        return torch.Generator().manual_seed(seed)

    def test_production_initialization_keeps_zero_output_and_adaln(self):
        model = self.model(nonzero=False)
        for name, parameter in model.energy_head.named_parameters():
            if 'modulation' in name or name.startswith('output.'):
                self.assertEqual(float(parameter.abs().sum()), 0.0, name)

    def test_energy_loss_subtracts_repulsion_and_has_expected_gradient(self):
        first = torch.tensor([[2.0, 0.0]], requires_grad=True)
        second = torch.tensor([[0.0, 2.0]], requires_grad=True)
        target = torch.zeros_like(first)
        terms = energy_score_terms(first, second, target, temperature=0.75)
        expected = 4.0 - 0.75 * (8.0 ** 0.5)
        self.assertAlmostEqual(float(terms['loss']), expected, places=6)
        self.assertAlmostEqual(float(terms['repulsion']), 8.0 ** 0.5, places=6)
        terms['loss'].backward()
        expected_gradient = torch.tensor([[1 - 0.75 / 2**0.5, 0.75 / 2**0.5]])
        self.assertTrue(torch.allclose(first.grad, expected_gradient))

    def test_hidden_future_values_do_not_enter_conditioning(self):
        model = self.model().eval()
        prefix, controls, target = self.inputs()
        values = torch.cat((prefix, target), dim=1).requires_grad_()
        masked = torch.tensor([[False, False, False, True, False, True, True, False]]).expand(2, -1)
        dropped = torch.zeros(2, dtype=torch.bool)
        original = model._context(values, controls, masked, drop_labels=dropped,
                                  generator=self.rng(), training=False)
        changed = values.detach().clone()
        changed[masked] += 1000
        repeated = model._context(changed, controls, masked, drop_labels=dropped,
                                  generator=self.rng(), training=False)
        self.assertTrue(torch.equal(original, repeated))
        original.square().sum().backward()
        self.assertEqual(float(values.grad[masked].abs().sum()), 0.0)

    def test_training_dropout_and_sampling_use_only_supplied_rng(self):
        model = self.model().train()
        prefix, controls, target = self.inputs()
        global_before = torch.random.get_rng_state().clone()
        first = model.loss(target, prefix, controls, generator=self.rng())
        second = model.loss(target, prefix, controls, generator=self.rng())
        for key in ('loss', 'attraction', 'repulsion'):
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertGreater(float(first['repulsion']), 0.0)
        first['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        prefix_before = prefix.clone()
        sampled = model.sample(prefix, controls, 5, generator=self.rng(), steps=3)
        repeated = model.sample(prefix, controls, 5, generator=self.rng(), steps=3)
        self.assertTrue(torch.equal(sampled, repeated))
        self.assertTrue(torch.equal(prefix, prefix_before))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        self.assertTrue(model.training)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA allocation required')
    def test_cuda_generator_device_alias_preserves_loss_and_sample_replay(self):
        model = self.model().cuda()
        prefix, controls, target = (value.cuda() for value in self.inputs())
        before = torch.cuda.get_rng_state().clone()
        terms = model.loss(target, prefix, controls,
                           generator=torch.Generator(device='cuda').manual_seed(42))
        terms['loss'].backward()
        first = model.sample(prefix, controls, 5, steps=3,
                             generator=torch.Generator(device='cuda').manual_seed(43))
        second = model.sample(prefix, controls, 5, steps=3,
                              generator=torch.Generator(device=prefix.device).manual_seed(43))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))

    def test_shared_batch_mask_ratio_preserves_default_loss_and_rng(self):
        model = self.model().train()
        prefix, controls, target = self.inputs()
        original_rng = self.rng()
        original = model.loss(target, prefix, controls, generator=original_rng)
        shared_rng = self.rng()
        ratio = model.mask_ratio_min + (model.mask_ratio_max - model.mask_ratio_min) * float(
            torch.rand((), generator=shared_rng))
        shared = model.loss(target, prefix, controls, generator=shared_rng, mask_ratio=ratio)
        for key in ('loss', 'attraction', 'repulsion'):
            self.assertTrue(torch.equal(original[key], shared[key]))
        self.assertEqual(original['masked_tokens'], shared['masked_tokens'])
        self.assertTrue(torch.equal(original_rng.get_state(), shared_rng.get_state()))
        for value in (0.6, 1.1, True, float('nan')):
            with self.subTest(mask_ratio=value), self.assertRaises(ValueError):
                model.loss(target, prefix, controls, generator=self.rng(), mask_ratio=value)

    def test_empty_and_different_prefixes_preserve_control_information(self):
        model = self.model().eval()
        for past in (0, 2, 5):
            with self.subTest(past=past):
                prefix, controls, _ = self.inputs(past=past)
                left = model.sample(prefix, controls, 5, generator=self.rng(), steps=3, guidance=1)
                right = model.sample(prefix, controls + 2, 5, generator=self.rng(), steps=3, guidance=1)
                self.assertEqual(left.shape, (2, 5, 2))
                self.assertGreater(float((left - right).abs().max()), 0.0)

    def test_each_future_position_is_committed_exactly_once(self):
        model = self.model().eval()
        prefix, controls, _ = self.inputs()
        positions_seen = []
        original = model._predict_positions

        def record(values, controls, masked, positions, **kwargs):
            self.assertFalse(masked[:, :3].any())
            self.assertTrue(masked.gather(1, positions).all())
            positions_seen.append(positions.clone())
            return original(values, controls, masked, positions, **kwargs)

        with patch.object(model, '_predict_positions', side_effect=record):
            model.sample(prefix, controls, 5, generator=self.rng(), steps=3)
        self.assertEqual(len(positions_seen), 3)
        positions = torch.cat(positions_seen, dim=1).sort(dim=1).values
        self.assertTrue(torch.equal(positions, torch.arange(3, 8).expand(2, -1)))

    def test_guidance_interpolates_features_before_nonlinear_generator(self):
        model = self.model().eval()
        prefix, controls, target = self.inputs()
        values = torch.cat((prefix, target), dim=1)
        masked = torch.ones(2, 8, dtype=torch.bool)
        positions = torch.tensor([[3], [3]])

        class NonlinearHead(nn.Module):
            def forward(self, context, noise, **kwargs):
                return context[..., :2].square() + noise[..., :2]

        def context(values, controls, masked, *, drop_labels, **kwargs):
            level = torch.where(drop_labels, 1.0, 2.0)
            return level[:, None, None].expand(2, 8, 8)

        model.energy_head = NonlinearHead()
        with patch.object(model, '_context', side_effect=context):
            got = model._predict_positions(values, controls, masked, positions,
                generator=self.rng(), guidance=3.0, temperature=0.7)
        noise = torch.rand((2, 1, 4), generator=self.rng()) - 0.5
        expected = 16.0 + noise[..., :2]  # (1 + 3 * (2 - 1)) squared, not 1 + 3 * (4 - 1).
        self.assertTrue(torch.equal(got, expected))

    def test_linear_guidance_uses_source_remaining_mask_schedule(self):
        model = self.model().eval()
        prefix, controls, _ = self.inputs()
        actual = []

        def predict(values, controls, masked, positions, **kwargs):
            actual.append(kwargs['guidance'])
            return values.new_zeros(*positions.shape, model.latent_channels)

        with patch.object(model, '_predict_positions', side_effect=predict):
            model.sample(prefix, controls, 5, generator=self.rng(), steps=3, guidance=4)
        self.assertEqual(len(actual), 3)
        for found, expected in zip(actual, (1.6, 2.8, 3.4)):
            self.assertAlmostEqual(found, expected)

    def test_inference_temperature_scales_only_residual_noise_shift(self):
        torch.manual_seed(41)
        block = _AdaLNBlock(3).eval()
        self.activate(block)
        value = torch.tensor([[[1.0, -2.0, 0.5]]])
        noise = torch.tensor([[[0.2, 0.7, -0.3]]])
        shift, scale, gate = block.modulation(F.silu(noise)).chunk(3, dim=-1)
        expected = value + gate * block.linear_out(F.silu(
            block.linear_in(block.norm(value) * (1 + scale) + 0.3 * shift)))
        got = block(value, noise, training=False, temperature=0.3)
        self.assertTrue(torch.equal(got, expected))
        wrong = value + 0.3 * gate * block.linear_out(F.silu(
            block.linear_in(block.norm(value) * (1 + 0.3 * scale) + 0.3 * shift)))
        self.assertFalse(torch.allclose(got, wrong))
        head = _EnergyHead(3, 2, 3, 0, 3).eval()
        self.activate(head)
        first = head(value, noise, training=False, temperature=0.3)
        second = head(value, noise, training=False, temperature=1.0)
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(float(first.abs().sum()), 0.0)

    def test_state_dict_restores_sampling_and_rejects_changed_protocol(self):
        model = self.model().eval()
        prefix, controls, _ = self.inputs()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        first = model.sample(prefix, controls, 5, generator=self.rng(), steps=3)
        second = restored.sample(prefix, controls, 5, generator=self.rng(), steps=3)
        self.assertTrue(torch.equal(first, second))
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.model(label_dropout=0.2).load_state_dict(model.state_dict())

    def test_invalid_budget_inputs_and_response_free_sample_interface(self):
        model = self.model()
        prefix, controls, target = self.inputs()
        with self.assertRaisesRegex(ValueError, 'steps'):
            model.sample(prefix, controls, 5, generator=self.rng(), steps=6)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            model.sample(prefix, controls[:, :-1], 5, generator=self.rng(), steps=3)
        with self.assertRaisesRegex(ValueError, 'finite'):
            model.loss(target * float('nan'), prefix, controls, generator=self.rng())
        for future, steps in ((1, 1), (5, 3), (100, 50)):
            commits = cosine_commit_counts(future, steps)
            self.assertEqual(len(commits), steps)
            self.assertEqual(sum(commits), future)
            self.assertTrue(all(count > 0 for count in commits))
        self.assertEqual(set(inspect.signature(model.sample).parameters),
                         {'prefix', 'controls', 'future_bins', 'generator', 'steps', 'guidance', 'temperature'})


if __name__ == '__main__':
    unittest.main()
