"""Native observed-path mixture ordering, gradients and explicit RNG contracts."""
import math
from unittest.mock import patch

import pytest
import torch

from .population_observation_prior import prior_observation_mixture_loss
from .population_prior_objective import PRIOR_PARAMETER_NAMES, configure_prior_parameters
from .test_population_predictive_mixture import example


def _streams(seed=41):
    seeds = (41, 42, 43)
    return tuple(torch.Generator().manual_seed(value)
                 for value in (seed, seeds[(seeds.index(seed) + 1) % len(seeds)]))


def _arguments(dtype=torch.float64):
    torch.set_num_threads(1)
    model = example().to(dtype=dtype)
    configure_prior_parameters(model)
    data = torch.Generator().manual_seed(43)
    mean = torch.randn(2, 10, 3, 2, dtype=dtype, generator=data)
    variance = .1 + torch.rand(mean.shape, dtype=dtype, generator=data)
    controls = torch.linspace(-1, 1, 20, dtype=dtype).reshape(2, 10, 1)
    counts = torch.zeros(2, 2, 4, len(model.unit_ids), dtype=torch.int64)
    # Events only at the final bin's final tick admit full eligible exposure.
    counts[:, :, -1, ::2] = 1
    eligible = torch.full_like(counts, model.bin_ticks)
    return dict(model=model, mean=mean, variance=variance, controls=controls,
                target_counts=counts, target_eligible_ticks=eligible,
                origins=(2, 5), horizons=(1, 2, 4))


def _loss(arguments, *, streams=None, **changes):
    initial, dynamics = _streams() if streams is None else streams
    return prior_observation_mixture_loss(
        **dict(arguments, initial_generator=initial, dynamics_generator=dynamics, **changes))


def _manual_replay(arguments, streams, particles=8):
    """Independent batch/origin axes and per-unit event likelihood formula."""
    model = arguments["model"]
    mean, variance, controls = (arguments[key].detach() for key in ("mean", "variance", "controls"))
    origins, horizons = arguments["origins"], arguments["horizons"]
    boundary = [origin - 1 for origin in origins]
    shape = (particles, mean.shape[0], len(origins), *mean.shape[2:])
    state = mean[:, boundary] + variance[:, boundary].sqrt() * torch.randn(
        shape, dtype=mean.dtype, generator=streams[0])
    states = []
    for step in range(1, max(horizons) + 1):
        indices = [origin + step - 1 for origin in origins]
        state = model.transition_mean(state, controls[:, indices]) + model.process_log_variance.exp().sqrt() * torch.randn(
            shape, dtype=mean.dtype, generator=streams[1])
        states.append(state)
    trajectory = torch.stack(states, dim=3)
    log_rate = model.log_hazard(trajectory)
    hazard = log_rate.exp() * model.tick_seconds
    counts = arguments["target_counts"].to(mean.dtype)
    eligible = arguments["target_eligible_ticks"].to(mean.dtype)
    per_unit = (eligible - counts) * hazard - counts * torch.log(-torch.expm1(-hazard))
    terms = {}
    for horizon in horizons:
        joint = per_unit[..., :horizon, :].sum((-1, -2))
        terms[f"prior_observation_nll_h{horizon}"] = (
            math.log(particles) - torch.logsumexp(-joint, 0)).mean() / (len(model.unit_ids) * horizon)
        weights = torch.softmax(-joint.detach(), dim=0)
        terms[f"prior_observation_particle_ess_h{horizon}"] = weights.square().sum(0).reciprocal().mean()
    terms["loss"] = torch.stack([terms[f"prior_observation_nll_h{h}"] for h in horizons]).mean()
    return terms, trajectory


def _equal_terms(actual, expected, *, rtol=1e-11, atol=1e-12):
    assert set(actual) == set(expected)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=rtol, atol=atol)


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_native_path_manual_replay_final_noise_and_batch_major_cases(seed):
    arguments = _arguments()
    arguments.update(origins=(5, 2), horizons=(4, 1, 2))
    actual_streams, expected_streams = _streams(seed), _streams(seed)
    global_before = torch.get_rng_state().clone()
    observed = []
    native_readout = arguments["model"].log_hazard

    def readout(trajectory, start=0, stop=None):
        observed.append(trajectory.detach().clone())
        return native_readout(trajectory, start, stop)

    with patch.object(arguments["model"], "log_hazard", side_effect=readout):
        actual = _loss(arguments, streams=actual_streams, unit_chunk_size=3)
    expected, trajectory = _manual_replay(arguments, expected_streams)
    _equal_terms(actual, expected)
    for path in observed:
        torch.testing.assert_close(path, trajectory.reshape(8, 4, 4, 3, 2), rtol=0, atol=0)
    for actual_generator, expected_generator in zip(actual_streams, expected_streams):
        assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
    assert torch.equal(global_before, torch.get_rng_state())
    for key, value in actual.items():
        if "_ess_" in key:
            assert not value.requires_grad and 1 <= float(value) <= 8


def test_mixture_must_cover_all_units_and_the_entire_time_prefix():
    arguments = _arguments()
    model = arguments["model"]
    units = len(model.unit_ids)
    arguments.update(mean=arguments["mean"][:1], variance=arguments["variance"][:1],
                     controls=arguments["controls"][:1], origins=(2,), horizons=(2,),
                     target_counts=torch.zeros(1, 1, 2, units, dtype=torch.int64),
                     target_eligible_ticks=torch.full((1, 1, 2, units), model.bin_ticks, dtype=torch.int64))
    parity = (torch.arange(2)[:, None, None, None] + torch.arange(2)[None, None, :, None]
              + torch.arange(units)[None, None, None, :]) % 2
    hazard = torch.where(parity == 0, 3.0, .1).double() / model.bin_ticks
    log_rate = torch.log(hazard / model.tick_seconds)
    with patch.object(model, "log_hazard", side_effect=lambda trajectory, start=0, stop=None: log_rate[..., start:stop]):
        actual = _loss(arguments, prediction_particles=2, unit_chunk_size=2)
    per_unit = model.bin_ticks * hazard
    joint = per_unit.sum((-1, -2))
    expected = (math.log(2) - torch.logsumexp(-joint, 0)).mean() / (units * 2)
    wrong = (math.log(2) - torch.logsumexp(-per_unit, 0)).sum((-1, -2)).mean() / (units * 2)
    torch.testing.assert_close(actual["loss"], expected, rtol=1e-12, atol=1e-12)
    assert float(expected - wrong) > .1
    torch.testing.assert_close(actual["prior_observation_particle_ess_h2"], torch.tensor(2., dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_unit_chunk_loss_and_all_prior_gradients_agree(dtype):
    arguments = _arguments(dtype)
    parameters = tuple(getattr(arguments["model"], name) for name in PRIOR_PARAMETER_NAMES)
    reference = _loss(arguments, unit_chunk_size=len(arguments["model"].unit_ids))
    reference_gradients = torch.autograd.grad(reference["loss"], parameters)
    rtol, atol = (2e-5, 1e-7) if dtype == torch.float32 else (1e-11, 1e-12)
    for chunk in (1, 3, None):
        actual = _loss(arguments, unit_chunk_size=chunk)
        gradients = torch.autograd.grad(actual["loss"], parameters)
        _equal_terms(actual, reference, rtol=rtol, atol=atol)
        for gradient, expected in zip(gradients, reference_gradients):
            torch.testing.assert_close(gradient, expected, rtol=rtol, atol=atol)


def test_only_origin_boundaries_use_posterior_and_controls_keep_native_endpoints():
    arguments = _arguments()
    reference = _loss(arguments)
    changed = dict(arguments)
    changed["mean"], changed["variance"] = arguments["mean"].clone(), arguments["variance"].clone()
    unused = [index for index in range(10) if index not in (1, 4)]
    changed["mean"][:, unused] += 100
    changed["variance"][:, unused] += 10
    changed["controls"] = arguments["controls"].clone()
    changed["controls"][:, [0, 1, 9]] += 100
    _equal_terms(_loss(changed), reference, rtol=0, atol=0)
    changed["controls"][:, 8] += 20
    result = _loss(changed)
    assert not torch.equal(result["prior_observation_nll_h4"], reference["prior_observation_nll_h4"])
    assert torch.equal(result["prior_observation_nll_h1"], reference["prior_observation_nll_h1"])


def test_changed_event_labels_do_not_change_latent_paths_or_random_streams():
    arguments = _arguments()
    paths = []
    native_readout = arguments["model"].log_hazard

    def readout(trajectory, start=0, stop=None):
        paths.append(trajectory.detach().clone())
        return native_readout(trajectory, start, stop)

    first, second = _streams(), _streams()
    with patch.object(arguments["model"], "log_hazard", side_effect=readout):
        original = _loss(arguments, streams=first, unit_chunk_size=len(arguments["model"].unit_ids))
        counts = arguments["target_counts"].clone()
        exposure = arguments["target_eligible_ticks"].clone()
        counts[:, :, -1, 1] = 1
        exposure[:, :, 0] -= 1
        changed = _loss(arguments, streams=second, target_counts=counts, target_eligible_ticks=exposure,
                        unit_chunk_size=len(arguments["model"].unit_ids))
    assert len(paths) == 2 and torch.equal(paths[0], paths[1])
    assert not torch.equal(original["loss"], changed["loss"])
    for left, right in zip(first, second):
        assert torch.equal(left.get_state(), right.get_state())


def test_frozen_readout_retains_latent_gradients_and_only_six_prior_tensors_update():
    arguments = _arguments()
    model = arguments["model"]
    for name in ("mean", "variance", "controls", "target_counts", "target_eligible_ticks"):
        arguments[name] = arguments[name].to(torch.float64).detach().requires_grad_()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    parameters = tuple(getattr(model, name) for name in PRIOR_PARAMETER_NAMES)
    optimizer = torch.optim.SGD(parameters, lr=.001)
    with patch.object(model, "encode", side_effect=AssertionError("posterior must remain fixed")):
        result = _loss(arguments)
        result["loss"].backward()
    for name in ("mean", "variance", "controls", "target_counts", "target_eligible_ticks"):
        assert arguments[name].grad is None, name
    for name, parameter in model.named_parameters():
        if name in PRIOR_PARAMETER_NAMES:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert float(parameter.grad.abs().sum()) > 0, name
        else:
            assert not parameter.requires_grad and parameter.grad is None, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]) == (name not in PRIOR_PARAMETER_NAMES), name


def test_stream_restore_and_one_step_both_consume_native_dynamics_without_global_rng():
    arguments = _arguments()
    arguments.update(horizons=(1,), target_counts=arguments["target_counts"][:, :, :1],
                     target_eligible_ticks=arguments["target_eligible_ticks"][:, :, :1])
    streams = _streams()
    global_before = torch.get_rng_state().clone()
    dynamics_before = streams[1].get_state().clone()
    first = _loss(arguments, streams=streams)
    assert not torch.equal(dynamics_before, streams[1].get_state())
    committed = tuple(generator.get_state().clone() for generator in streams)
    second = _loss(arguments, streams=streams)
    assert not torch.equal(first["loss"], second["loss"])
    after = tuple(generator.get_state().clone() for generator in streams)
    for generator, state in zip(streams, committed):
        generator.set_state(state)
    _equal_terms(_loss(arguments, streams=streams), second, rtol=0, atol=0)
    for generator, state in zip(streams, after):
        assert torch.equal(generator.get_state(), state)
    assert torch.equal(global_before, torch.get_rng_state())


def test_invalid_labels_rejected_before_random_streams_advance():
    arguments = _arguments()
    counts, exposure = arguments["target_counts"], arguments["target_eligible_ticks"]
    invalid = [dict(target_counts=None), dict(target_counts=counts[:, :1]),
               dict(target_counts=counts[..., :-1]), dict(target_counts=counts.to("meta")),
               dict(target_counts=counts.bool()), dict(target_counts=counts.to(torch.complex128)),
               dict(target_counts=counts.double() + .5), dict(target_counts=-torch.ones_like(counts)),
               dict(target_counts=torch.full_like(counts.double(), float("nan"))),
               dict(target_eligible_ticks=exposure[:, :, :-1]),
               dict(target_eligible_ticks=counts - 1), dict(target_eligible_ticks=exposure + 1),
               dict(target_eligible_ticks=exposure.double() - .5),
               dict(target_eligible_ticks=torch.full_like(exposure.double(), float("inf")))]
    for change in invalid:
        streams = _streams()
        before = tuple(generator.get_state().clone() for generator in streams)
        with pytest.raises(ValueError, match="labels|integer|native bin"):
            _loss(arguments, streams=streams, **change)
        for generator, state in zip(streams, before):
            assert torch.equal(generator.get_state(), state)


def test_invalid_streams_shapes_particle_counts_and_parameter_profile_rejected():
    arguments = _arguments()
    shared = torch.Generator().manual_seed(41)
    for streams in ((None, shared), (shared, None), (shared, shared),
                    (torch.Generator().manual_seed(42), torch.Generator().manual_seed(42))):
        before = tuple(generator.get_state().clone() if generator is not None else None for generator in streams)
        with pytest.raises(ValueError, match="random streams"):
            _loss(arguments, streams=streams)
        for generator, state in zip(streams, before):
            if generator is not None:
                assert torch.equal(generator.get_state(), state)
    invalid = [dict(prediction_particles=value) for value in (0, -1, True, 1.5, None)]
    invalid += [dict(unit_chunk_size=value) for value in (0, -1, True, 1.5)]
    invalid += [dict(origins=(2, 2)), dict(horizons=(0,)), dict(origins=(8,)),
                dict(variance=torch.zeros_like(arguments["variance"])),
                dict(controls=arguments["controls"].float())]
    for change in invalid:
        streams = _streams()
        before = tuple(generator.get_state().clone() for generator in streams)
        with pytest.raises(ValueError):
            _loss(arguments, streams=streams, **change)
        for generator, state in zip(streams, before):
            assert torch.equal(generator.get_state(), state)
    arguments["model"].unit_loading.requires_grad_(True)
    with pytest.raises(ValueError, match="six native prior parameters"):
        _loss(arguments)


def test_nonfinite_process_variance_rejected_without_random_draws():
    arguments = _arguments()
    with torch.no_grad():
        arguments["model"].process_log_variance.fill_(float("-inf"))
    streams = _streams()
    before = tuple(generator.get_state().clone() for generator in streams)
    with pytest.raises(FloatingPointError, match="process variance"):
        _loss(arguments, streams=streams)
    for generator, state in zip(streams, before):
        assert torch.equal(generator.get_state(), state)
