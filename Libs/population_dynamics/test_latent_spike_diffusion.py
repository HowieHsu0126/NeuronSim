"""Meaningful prefix, RNG, gradient and identity checks for adapted LDNS."""
import io
import unittest

import torch

from .latent_spike_diffusion import ControlledLatentSpikeDiffusion,PrefixCondition


class LatentSpikeDiffusionChecks(unittest.TestCase):
    def model(self):
        torch.manual_seed(41)
        model = ControlledLatentSpikeDiffusion(tuple(range(8)),input_size=1,bin_seconds=.005,
            latent_channels=8,hidden_channels=32,encoder_blocks=1,denoiser_blocks=1,diffusion_steps=8)
        model.set_latent_scale(torch.zeros(1,8,1),torch.ones(1,8,1))
        return model

    def test_training_and_joint_rng_replay(self):
        torch.set_num_threads(1)
        model = self.model()
        counts = torch.randint(0,3,(1,16,8)).float()
        mask = torch.rand(1,8,16)>.5
        terms = model.autoencoder_loss(counts,mask=mask)
        terms['loss'].backward()
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))
        state = model.infer_state(counts)
        target = model.encode_target(counts[:,:8])
        stimulus = torch.zeros(1,8,1)
        loss = model.diffusion_loss(target,state,stimulus,noise=torch.randn_like(target),
                                    timestep=torch.tensor([3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        before = torch.random.get_rng_state().clone()
        def generate():
            return model.simulate(state,stimulus,latent_rng=torch.Generator().manual_seed(12),
                                  observation_rng=torch.Generator().manual_seed(13))
        first,second = generate(),generate()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        for key in ['counts','latent','expected_counts']:
            self.assertTrue(torch.equal(first[key],second[key]),key)
        self.assertEqual(first['counts'].shape,(1,8,8))
        self.assertTrue(all(block.bidirectional is True for block in model.denoiser.blocks))

    def test_prefix_has_no_future_truth_or_gradient(self):
        model = self.model()
        episode = torch.randint(0,3,(1,32,8)).float().requires_grad_()
        first = model.infer_state(episode[:,:16])
        changed = episode.detach().clone();changed[:,16:]+=10
        second = model.infer_state(changed[:,:16])
        self.assertTrue(torch.equal(first.latent,second.latent))
        self.assertFalse(first.latent.requires_grad)
        # Source whole-episode normalization is not substituted for this prefix.
        with torch.no_grad():
            full1=model.autoencoder.encode(episode.transpose(1,2))[:,:,:16]
            full2=model.autoencoder.encode(changed.transpose(1,2))[:,:,:16]
        self.assertGreater(float((full1-full2).abs().max()),0.)

    def test_future_stimulus_has_gradient_and_identity_is_checked(self):
        model=self.model();state=model.infer_state(torch.ones(1,16,8))
        stimulus=torch.zeros(1,8,1,requires_grad=True)
        predicted=model._noise_prediction(torch.randn(1,8,8),torch.tensor([3]),state,stimulus)
        predicted.square().sum().backward()
        self.assertTrue(torch.isfinite(stimulus.grad).all())
        self.assertGreater(float(stimulus.grad.abs().sum()),0.)
        foreign=PrefixCondition(state.latent,tuple(range(1,9)),16)
        with self.assertRaisesRegex(ValueError,'matched observed prefix'):
            model._noise_prediction(torch.ones(1,8,8),torch.tensor([3]),foreign,stimulus)

    def test_checkpoint_restores_scale_and_rejects_foreign_map(self):
        model=self.model();buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        state=torch.load(buffer,weights_only=False);restored=self.model();restored.scale_fitted=False
        restored.load_state_dict(state);self.assertTrue(restored.scale_fitted)
        foreign=ControlledLatentSpikeDiffusion(tuple(range(1,9)),input_size=1,bin_seconds=.005,
            latent_channels=8,hidden_channels=32,encoder_blocks=1,denoiser_blocks=1,diffusion_steps=8)
        with self.assertRaisesRegex(ValueError,'checkpoint source'):
            foreign.load_state_dict(state)


if __name__=='__main__':
    unittest.main()
