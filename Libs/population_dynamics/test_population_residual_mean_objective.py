"""Native-term/RNG parity and finite-K-corrected field mean gradients."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .lif_neural_field import sampled_mean_field_mse
from .native_renewal_expectation import expected_field_counts, expected_sequence_counts
from .population_residual_mean_objective import field_mean_mse_terms, residual_mean_objective
from .population_residual_observation import configure_correction_parameters, residual_observation_mixture_loss
from .test_population_residual_observation import arguments, streams
from .test_population_native_mean_benchmark import inputs as readout_inputs


def combined_arguments(mode="history", seed=41, dtype=torch.float64):
    args = arguments(mode, seed, dtype=dtype)
    args["origins"] = (5, 2, 3)
    for name in ("target_counts", "target_eligible_ticks"):
        args[name] = torch.cat((args[name], args[name][:, :1]), dim=1)
    batch, origins, _, units = args["target_counts"].shape
    args["initial_ready_in_ticks"] = torch.arange(batch * origins * units).reshape(batch, origins, units) % args["model"].minimum_interval_ticks
    args["field_standard_deviation_hz"] = torch.linspace(.8, 1.3, len(args["model"].field_names), dtype=dtype)
    return args


def combined(args, seed=41, **options):
    initial, dynamics = streams(seed)
    result = residual_mean_objective(**args, initial_generator=initial, dynamics_generator=dynamics, **options)
    return result, (initial, dynamics)


@pytest.mark.parametrize("mode", ["current", "history"])
@pytest.mark.parametrize("seed", [41, 42, 43])
def test_native_terms_public_rng_and_reconstructed_paths_are_exact(mode, seed):
    args = combined_arguments(mode, seed)
    native_args = {key: value for key, value in args.items()
                   if key not in ("initial_ready_in_ticks", "field_standard_deviation_hz")}
    initial, dynamics = streams(seed)
    original = residual_observation_mixture_loss(**native_args, initial_generator=initial, dynamics_generator=dynamics,
                                                unit_chunk_size=4096, checkpoint_readout=False)
    expected_rng = [generator.get_state().clone() for generator in (initial, dynamics)]
    global_rng = torch.get_rng_state().clone()
    paths = []
    readout = args["model"].parent.log_hazard
    def capture(path, start=0, stop=None):
        paths.append(path.detach().clone())
        return readout(path, start, stop)
    with patch.object(args["model"].parent, "log_hazard", side_effect=capture):
        result, generators = combined(args, seed, unit_chunk_size=4096, mean_unit_chunk_size=32768,
                                      checkpoint_readout=False)
    assert len(paths) == 2
    native_path, mean_path = paths
    assert native_path.shape[:2] == (8, 6)
    assert torch.equal(native_path.reshape(48, *native_path.shape[2:]), mean_path)
    torch.testing.assert_close(result["native_loss"], original["loss"], rtol=0, atol=0)
    for name, value in original.items():
        if name != "loss":
            torch.testing.assert_close(result[name], value, rtol=0, atol=0)
    torch.testing.assert_close(result["loss"], result["native_loss"] + result["field_mean_loss"], rtol=0, atol=0)
    assert all(torch.equal(generator.get_state(), expected) for generator, expected in zip(generators, expected_rng))
    assert torch.equal(global_rng, torch.get_rng_state())
    assert {f"field_mean_mse_h{h}" for h in args["horizons"]}.issubset(result)


def test_field_mean_formula_matches_existing_numpy_correction_and_retains_negative_values():
    truth = np.arange(16, dtype=np.int64).reshape(2, 4, 2) % 4 + 2
    samples = np.stack([truth + (-1 if draw % 2 else 1) for draw in range(8)])
    units, sd, horizons = np.array([2, 5]), np.array([.7, 1.3]), (4, 1, 2)
    expected = torch.tensor(samples, dtype=torch.float64, requires_grad=True)
    terms = field_mean_mse_terms(expected, torch.from_numpy(truth), torch.from_numpy(units),
                                 torch.from_numpy(sd), .005, horizons)
    values = []
    for horizon in horizons:
        reference = np.mean([sampled_mean_field_mse(samples[:, case, :horizon], truth[case, :horizon], sd, units, 5.)
                             for case in range(2)])
        assert float(terms[f"field_mean_mse_h{horizon}"].detach()) == pytest.approx(reference, rel=1e-12, abs=1e-12)
        assert reference < 0
        values.append(reference)
    assert float(terms["field_mean_loss"].detach()) == pytest.approx(np.mean(values), rel=1e-12, abs=1e-12)
    gradient = torch.autograd.grad(terms["field_mean_loss"], expected)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("native_chunk,mean_chunk", [(1, 1), (3, 2)])
def test_checkpoint_and_chunk_changes_preserve_combined_values_and_gradients(dtype, native_chunk, mean_chunk):
    args = combined_arguments(dtype=dtype)
    parameters = configure_correction_parameters(args["model"])
    expected, _ = combined(args, unit_chunk_size=4, mean_unit_chunk_size=4, checkpoint_readout=False)
    expected_gradients = torch.autograd.grad(expected["loss"], parameters)
    actual, generators = combined(args, unit_chunk_size=native_chunk, mean_unit_chunk_size=mean_chunk,
                                  checkpoint_readout=True)
    states = [generator.get_state().clone() for generator in generators]
    actual_gradients = torch.autograd.grad(actual["loss"], parameters)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=tolerance, atol=tolerance * .01)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance * .01)
        assert torch.isfinite(actual_gradient).all() and actual_gradient.norm() > 0
    assert all(torch.equal(generator.get_state(), state) for generator, state in zip(generators, states))


def test_future_exposure_changes_native_term_but_never_mean_availability():
    args = combined_arguments()
    expected, _ = combined(args, checkpoint_readout=False)
    changed = dict(args, target_eligible_ticks=args["target_eligible_ticks"] - 1)
    actual, _ = combined(changed, checkpoint_readout=False)
    assert not torch.equal(actual["native_loss"], expected["native_loss"])
    for name in expected:
        if name.startswith("field_mean_"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_shared_field_helper_preserves_distinct_readiness_for_each_flattened_path():
    parent, path, ready = readout_inputs()
    distinct = torch.stack((ready, parent.minimum_interval_ticks - 1))
    actual = expected_field_counts(parent, path, distinct, unit_chunk_size=2, checkpoint_readout=True)
    counts = expected_sequence_counts(parent.log_hazard(path), distinct, parent.minimum_interval_ticks,
                                     parent.bin_ticks, parent.tick_seconds)
    expected = torch.stack([counts[..., parent.unit_field_index == field].sum(-1)
                            for field in range(len(parent.field_names))], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    shared = expected_field_counts(parent, path, ready, unit_chunk_size=2, checkpoint_readout=False)
    torch.testing.assert_close(actual[0], shared[0], rtol=0, atol=0)
    assert not torch.equal(actual[1], shared[1])


def test_forward_backward_does_not_update_parameters_inputs_or_public_rng_beyond_native_call():
    args = combined_arguments()
    model = args["model"]
    frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}
    saved_inputs = {name: value.detach().clone() for name, value in args.items() if isinstance(value, torch.Tensor)}
    result, generators = combined(args)
    states = [generator.get_state().clone() for generator in generators]
    torch.autograd.grad(result["loss"], configure_correction_parameters(model))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())
    assert all(torch.equal(value, args[name]) for name, value in saved_inputs.items())
    assert all(torch.equal(generator.get_state(), state) for generator, state in zip(generators, states))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_no_grad_validation_uses_the_same_combined_objective_and_rng_layout():
    args = combined_arguments()
    expected, generators = combined(args, checkpoint_readout=False)
    states = [generator.get_state().clone() for generator in generators]
    with torch.no_grad():
        actual, generators = combined(args, checkpoint_readout=True)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert not actual[name].requires_grad
    assert all(torch.equal(generator.get_state(), state) for generator, state in zip(generators, states))


@pytest.mark.parametrize("failure", ["ready_shape", "ready_dtype", "negative_ready", "ready_at_interval",
    "scale_shape", "zero_scale", "nonfinite_scale", "trainable_scale", "mean_chunk", "one_particle"])
def test_invalid_new_inputs_reject_before_public_rng_advances(failure):
    args = combined_arguments()
    options = {}
    if failure == "ready_shape":
        args["initial_ready_in_ticks"] = args["initial_ready_in_ticks"][:, :1]
    elif failure == "ready_dtype":
        args["initial_ready_in_ticks"] = args["initial_ready_in_ticks"].double()
    elif failure == "negative_ready":
        args["initial_ready_in_ticks"][0, 0, 0] = -1
    elif failure == "ready_at_interval":
        args["initial_ready_in_ticks"][0, 0, 0] = args["model"].minimum_interval_ticks[0]
    elif failure == "scale_shape":
        args["field_standard_deviation_hz"] = args["field_standard_deviation_hz"][:1]
    elif failure == "zero_scale":
        args["field_standard_deviation_hz"][0] = 0
    elif failure == "nonfinite_scale":
        args["field_standard_deviation_hz"][0] = float("nan")
    elif failure == "trainable_scale":
        args["field_standard_deviation_hz"].requires_grad_()
    elif failure == "mean_chunk":
        options["mean_unit_chunk_size"] = 0
    else:
        options["prediction_particles"] = 1
    initial, dynamics = streams()
    states = [generator.get_state().clone() for generator in (initial, dynamics)]
    with pytest.raises(ValueError, match="(readiness|standard deviations|particles|unit chunk)"):
        residual_mean_objective(**args, initial_generator=initial, dynamics_generator=dynamics, **options)
    assert all(torch.equal(generator.get_state(), state) for generator, state in zip((initial, dynamics), states))
