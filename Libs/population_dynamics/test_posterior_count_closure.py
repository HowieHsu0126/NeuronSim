"""Check the diagnostic against the unmodified native population sampler."""
import pytest
import torch

from .population_state import ControlledPopulationState
from .poisson_state_space import LatentRandomStreams
from .posterior_count_closure import sample_prescribed_latents


@pytest.fixture(params=[41, 42, 43])
def native_case(request):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ControlledPopulationState(unit_ids=list(range(7)), unit_field_index=[0,1,0,1,0,1,0],
        field_names=["E", "I", "empty"], minimum_interval_ticks=[3]*7,
        tick_seconds=.001, bin_seconds=.005, input_size=1, latent_channels=2,
        encoder_size=5, initialization_seed=request.param).double().to(device)
    with torch.no_grad():
        model.log_hazard_bias.fill_(5.)
    ready = torch.tensor([[0,1,2,0,1,2,0], [2,1,0,2,1,0,2]], device=device)
    state = model.infer_state(dict(counts=torch.zeros(2,4,7,device=device,dtype=torch.float64),
        stimulus=torch.zeros(2,4,1,device=device,dtype=torch.float64), ready_in_ticks=ready))
    seeds = [41,42,43]; i=request.param-41
    def rng():
        return LatentRandomStreams.from_seeds(seeds[i],seeds[(i+1)%3],seeds[(i+2)%3],device=device)
    native = list(model.iter_simulation(state, torch.zeros(2,6,1,device=device,dtype=torch.float64),
                                       rng(), 16, record_events=False))
    return model, torch.stack([x.latent for x in native],2), ready, torch.stack([x.counts for x in native],2), native[-1].ready_in_ticks, rng


def test_exact_native_replay_and_immutable_inputs(native_case):
    model,z,ready,expected,terminal,rng = native_case
    before_z,before_ready=z.clone(),ready.clone()
    before={k:v.clone() for k,v in model.state_dict().items()}
    actual,end=sample_prescribed_latents(model,z,ready,rng().observation)
    assert torch.equal(actual,expected) and torch.equal(end,terminal)
    assert torch.equal(z,before_z) and torch.equal(ready,before_ready)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_endogenous_clock_continuation(native_case):
    model,z,ready,expected,terminal,rng = native_case
    generator=rng().observation
    first,boundary=sample_prescribed_latents(model,z[:,:,:3],ready,generator)
    second,end=sample_prescribed_latents(model,z[:,:,3:],boundary,generator)
    assert torch.equal(torch.cat([first,second],2),expected)
    assert torch.equal(end,terminal)


@pytest.mark.parametrize("failure",["shape","nan","float_clock","invalid_clock","empty_time"])
def test_invalid_input_rejected(native_case,failure):
    model,z,ready,_,_,rng=native_case
    z,ready=z.clone(),ready.clone()
    if failure=="shape": z=z[0]
    elif failure=="nan": z[0,0,0,0,0]=float("nan")
    elif failure=="float_clock": ready=ready.float()
    elif failure=="invalid_clock": ready[0,0]=3
    else: z=z[:,:,:0]
    with pytest.raises(ValueError): sample_prescribed_latents(model,z,ready,rng().observation)
