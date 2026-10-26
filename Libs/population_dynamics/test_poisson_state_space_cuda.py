"""Real CUDA device identity and replay of the supported stream factory."""
import unittest

import torch

from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PoissonCudaTests(unittest.TestCase):
    def test_unindexed_cuda_streams_bind_to_model_and_replay(self):
        model = ControlledPoissonLDS([1, 3], 1, 2, 3, .005, initialization_seed=41).cuda()
        history = {"counts": torch.zeros(1, 3, 2, device="cuda"),
                   "stimulus": torch.zeros(1, 3, 1, device="cuda")}
        state = model.infer_state(history)
        first = LatentRandomStreams.from_seeds(85001, 85002, 85003, device="cuda")
        self.assertEqual(first.initial.device, model.loading.device)
        draws = model.simulate(state, torch.zeros(1, 4, 1, device="cuda"), first, 3)
        replay = model.simulate(state, torch.zeros(1, 4, 1, device="cuda"),
                                LatentRandomStreams.from_seeds(85001, 85002, 85003, device="cuda"), 3)
        self.assertTrue(torch.equal(draws.counts, replay.counts))
        self.assertTrue(torch.equal(draws.latent_states, replay.latent_states))


if __name__ == "__main__":
    unittest.main()
