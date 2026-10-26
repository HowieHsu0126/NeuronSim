"""Native replay, gradients, causality, training update and export contracts."""
import math
from unittest.mock import patch
import numpy as np
import pytest
import torch
from .population_observation_prior import prior_observation_mixture_loss
from .population_prior_objective import configure_prior_parameters
from .population_residual_memory import ResidualMemoryBoundary, ResidualMemoryPopulationState
from .population_residual_observation import (configure_correction_parameters, export_affine_predictor,
    residual_observation_mixture_loss, standardized_ridge_penalty)
from .residual_prediction import RidgeModel
from .poisson_state_space import LatentRandomStreams
from .test_population_residual_memory import make_parent, predictor


def streams(seed=41):
    seeds = (41, 42, 43)
    return tuple(torch.Generator().manual_seed(s) for s in (seed, seeds[(seeds.index(seed) + 1) % 3]))


def arguments(mode="history", seed=41, dtype=torch.float64, steps=4):
    torch.set_num_threads(1)
    model = ResidualMemoryPopulationState(make_parent().to(dtype),
        predictor(mode, memory_gain=.15, constant=.01), mode, {})
    configure_correction_parameters(model)
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn(2, steps + 5, 2, 2, generator=generator, dtype=dtype) * .1
    counts = torch.zeros(2, 2, steps, 4, dtype=torch.int64)
    counts[:, :, -1, ::2] = 1
    return dict(model=model, mean=mean, variance=torch.full_like(mean, .05),
        controls=torch.linspace(-.2, .2, 2 * (steps + 5), dtype=dtype).reshape(2, steps + 5, 1),
        target_counts=counts, target_eligible_ticks=torch.full_like(counts, 5),
        origins=(5, 2), horizons=(steps, 1, 2))


def loss(args, seed=41, **options):
    initial, dynamics = streams(seed)
    return residual_observation_mixture_loss(**args, initial_generator=initial,
        dynamics_generator=dynamics, **options)


def sampler_paths(args, seed=41):
    model, origins = args["model"], args["origins"]
    m, v, u = (args[key] for key in ("mean", "variance", "controls"))
    boundary = m[:, [o - 1 for o in origins]].reshape(-1, 2, 2)
    before = m[:, [o - 2 for o in origins]].reshape(-1, 2, 2)
    memory = boundary - model.parent.transition_mean(before, u[:, [o - 1 for o in origins]].reshape(-1, 1))
    state = ResidualMemoryBoundary(boundary, v[:, [o - 1 for o in origins]].reshape_as(boundary),
        torch.zeros(len(boundary), 4, dtype=torch.long), model.unit_ids, model.field_names,
        model.bin_seconds, model.source_metadata_identity, memory=memory)
    scheduled = torch.stack([u[:, [o + t for o in origins]].reshape(-1, 1)
                             for t in range(max(args["horizons"]))], 1)
    initial, dynamics = streams(seed)
    observation = next(s for s in (41, 42, 43) if s not in (initial.initial_seed(), dynamics.initial_seed()))
    rng = LatentRandomStreams.from_seeds(initial.initial_seed(), dynamics.initial_seed(), observation, device="cpu")
    result = list(model.iter_simulation(state, scheduled, rng, 8, record_events=False))
    return torch.stack([r.latent for r in result], 2), rng


@pytest.mark.parametrize("mode", ["current", "history"])
@pytest.mark.parametrize("seed", [41, 42, 43])
def test_frozen_native_sampler_and_joint_likelihood_agree(mode, seed):
    args = arguments(mode, seed)
    trajectory, replay = sampler_paths(args, seed)
    initial, dynamics = streams(seed)
    global_before = torch.get_rng_state().clone()
    result = residual_observation_mixture_loss(**args, initial_generator=initial, dynamics_generator=dynamics)
    hazard = args["model"].parent.log_hazard(trajectory).exp() * args["model"].tick_seconds
    counts = args["target_counts"].reshape(4, 4, 4)
    eligible = args["target_eligible_ticks"].reshape_as(counts)
    per_unit = (eligible - counts) * hazard - counts * torch.log(-torch.expm1(-hazard))
    for horizon in args["horizons"]:
        joint = per_unit[:, :, :horizon].sum((-1, -2))
        expected = (math.log(8) - torch.logsumexp(-joint, 0)).mean() / (4 * horizon)
        torch.testing.assert_close(result[f"native_nll_h{horizon}"], expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(initial.get_state(), replay.initial.get_state())
    assert torch.equal(dynamics.get_state(), replay.dynamics.get_state())
    assert torch.equal(global_before, torch.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_checkpoint_chunks_preserve_gradients_and_random_streams(dtype):
    args = arguments(dtype=dtype)
    params = configure_correction_parameters(args["model"])
    expected = loss(args, unit_chunk_size=4, checkpoint_readout=False)
    gradients = torch.autograd.grad(expected["loss"], params)
    for chunk in (1, 3):
        initial, dynamics = streams()
        actual = residual_observation_mixture_loss(**args, initial_generator=initial,
            dynamics_generator=dynamics, unit_chunk_size=chunk)
        states = [g.get_state().clone() for g in (initial, dynamics)]
        actual_gradients = torch.autograd.grad(actual["loss"], params)
        tolerance = 2e-5 if dtype == torch.float32 else 1e-11
        for a, b in zip(actual_gradients, gradients):
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance * .01)
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=tolerance, atol=tolerance * .01)
        assert all(torch.equal(g.get_state(), s) for g, s in zip((initial, dynamics), states))


def test_future_posteriors_and_event_labels_do_not_drive_latent_rollout():
    args = arguments()
    args.update(origins=(2,), target_counts=args["target_counts"][:, :1],
                target_eligible_ticks=args["target_eligible_ticks"][:, :1])
    original = loss(args)
    changed = dict(args, mean=args["mean"].clone(), variance=args["variance"].clone())
    changed["mean"][:, 2:] += 100
    changed["variance"][:, 2:] *= 10
    torch.testing.assert_close(loss(changed)["loss"], original["loss"], rtol=0, atol=0)
    paths = []
    readout = args["model"].parent.log_hazard
    def capture(z, start=0, stop=None):
        paths.append(z.detach().clone())
        return readout(z, start, stop)
    with patch.object(args["model"].parent, "log_hazard", side_effect=capture):
        loss(args, checkpoint_readout=False, unit_chunk_size=4)
        loss(dict(args, target_counts=torch.zeros_like(args["target_counts"])), checkpoint_readout=False, unit_chunk_size=4)
    assert torch.equal(paths[0], paths[1])
    changed = dict(args, controls=args["controls"].clone())
    changed["controls"][:, 5] += 2
    assert not torch.equal(loss(changed)["native_nll_h4"], original["native_nll_h4"])
    torch.testing.assert_close(loss(changed)["native_nll_h1"], original["native_nll_h1"], rtol=0, atol=0)


def test_zero_correction_matches_parent_native_objective():
    args = arguments("current")
    with torch.no_grad():
        args["model"].correction_weight.zero_()
        args["model"].correction_bias.zero_()
    expected = loss(args, checkpoint_readout=False)
    configure_prior_parameters(args["model"].parent)
    initial, dynamics = streams()
    original = prior_observation_mixture_loss(**dict(args, model=args["model"].parent),
        initial_generator=initial, dynamics_generator=dynamics)
    torch.testing.assert_close(expected["loss"], original["loss"], rtol=1e-12, atol=1e-12)


def test_100_step_update_preserves_parent_and_exports_to_native_runtime():
    args = arguments(steps=100)
    model = args["model"]
    reference = predictor("history", memory_gain=.15, constant=.01)
    frozen = {k: v.clone() for k, v in model.parent.state_dict().items()}
    params = configure_correction_parameters(model)
    optimizer = torch.optim.SGD(params, lr=1e-4)
    objective = loss(args)["loss"] + .001 * standardized_ridge_penalty(model, reference)
    objective.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert all(p.grad is None for p in model.parent.parameters())
    optimizer.step()
    assert float(loss(args)["loss"] + .001 * standardized_ridge_penalty(model, reference)) < float(objective)
    assert all(torch.equal(v, model.parent.state_dict()[k]) for k, v in frozen.items())
    exported = export_affine_predictor(model, reference)
    restored = ResidualMemoryPopulationState(make_parent().double(), exported, "history", {"objective": "native"})
    restored.parent.load_state_dict(model.parent.state_dict())
    x = np.linspace(-.5, .5, 18).reshape(2, 9)
    expected = model.correction(torch.from_numpy(x[:, :4].reshape(2, 2, 2)),
        torch.from_numpy(x[:, 4:5]), torch.from_numpy(x[:, 5:].reshape(2, 2, 2))).detach().numpy()
    np.testing.assert_allclose(exported.predict(x).reshape(2, 2, 2), expected, rtol=1e-12, atol=1e-12)
    a, _ = sampler_paths(args)
    b, _ = sampler_paths(dict(args, model=restored))
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_nontrivial_fitting_scales_preserve_penalty_and_export():
    rng = np.random.default_rng(43)
    reference = RidgeModel(1., rng.normal(size=9), rng.uniform(.5, 2, 9),
        rng.normal(size=4), rng.uniform(.5, 2, 4), rng.normal(size=(9, 4)) * .02)
    model = ResidualMemoryPopulationState(make_parent().double(), reference, "history", {})
    configure_correction_parameters(model)
    assert float(standardized_ridge_penalty(model, reference)) == pytest.approx(np.square(reference.coefficients).sum() / 4, rel=1e-12)
    with torch.no_grad(): model.correction_bias.add_(.05)
    x = rng.normal(size=(5, 9))
    np.testing.assert_allclose(export_affine_predictor(model, reference).predict(x),
        reference.predict(x) + .05, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("failure", ["origin", "labels", "parameter", "rng"])
def test_invalid_inputs_reject_before_random_draws(failure):
    args = arguments()
    initial, dynamics = streams()
    if failure == "origin": args["origins"] = (1, 2)
    if failure == "labels": args["target_eligible_ticks"][0, 0, 0, 0] = 6
    if failure == "parameter": args["model"].parent.log_hazard_bias.requires_grad_(True)
    if failure == "rng": dynamics = initial
    before = initial.get_state().clone()
    with pytest.raises(ValueError):
        residual_observation_mixture_loss(**args, initial_generator=initial, dynamics_generator=dynamics)
    assert torch.equal(before, initial.get_state())
