"""Analytic source-field reconstruction and deterministic response checks."""
import numpy as np
import pytest
import torch

from .population_activity_diagnostic import posterior_activity_bounds
from .posterior_field_reconstruction import posterior_field_reconstruction, response_covariance


class FrozenReadout(torch.nn.Module):
    unit_ids = (101, 7, 209, 3, 55)
    field_names = ("A", "empty", "B")
    latent_channels = 2
    tick_seconds = .01
    bin_ticks = 4
    bin_seconds = .04

    def __init__(self):
        super().__init__()
        self.register_buffer("unit_field_index", torch.tensor([2, 0, 2, 0, 2]))
        self.register_buffer("unit_loading", torch.tensor(
            [[1., .5], [-.5, 1.], [.25, -1.], [1.5, .75], [-1., .5]], dtype=torch.float64))
        self.register_buffer("bias", torch.tensor([.5, 1., 2., 3., 4.], dtype=torch.float64).log())

    def log_hazard(self, latent, start=0, stop=None):
        selected = latent.index_select(-2, self.unit_field_index[start:stop])
        return (selected * self.unit_loading[start:stop]).sum(-1) + self.bias[start:stop]


def inputs():
    mean = torch.arange(18, dtype=torch.float64).reshape(1, 3, 3, 2) / 100
    variance = torch.linspace(.01, .18, 18, dtype=torch.float64).reshape_as(mean)
    exposure = np.array([[4, 3, 0, 2, 1], [0, 0, 0, 0, 0], [1, 4, 2, 0, 3]])
    return mean, variance, exposure


def test_field_sums_reproduce_existing_posterior_activity_bounds():
    model = FrozenReadout()
    mean, variance, exposure = inputs()
    result = posterior_field_reconstruction(model, mean, variance, exposure, unit_chunk_size=2)
    reference = posterior_activity_bounds(model, mean, variance, exposure, unit_chunk_size=2)
    for field_key, aggregate_key in (
        ("posterior_integrated_intensity", "posterior_integrated_intensity"),
        ("posterior_expected_count_lower", "observed_path_expectation_lower"),
        ("posterior_expected_count_upper", "observed_path_expectation_upper"),
    ):
        np.testing.assert_allclose(result[field_key].sum(-1), reference[aggregate_key], rtol=1e-12, atol=1e-12)
    log_mean = model.log_hazard(mean)[0].numpy()
    mean_state = (-np.expm1(-np.exp(log_mean) * model.tick_seconds) * exposure).sum(-1)
    np.testing.assert_allclose(result["mean_state_count"].sum(-1), mean_state, rtol=1e-12, atol=1e-12)
    assert all(value.shape == (3, 3) and value.dtype == np.float64 for value in result.values())
    assert all(np.array_equal(value[:, 1], np.zeros(3)) for value in result.values())
    assert np.array_equal(result["mean_state_count"][1], np.zeros(3))
    assert np.all(result["posterior_integrated_intensity"][1, [0, 2]] > 0)


def test_zero_variance_closed_form_and_noncontiguous_source_field_mapping():
    model = FrozenReadout()
    mean = torch.zeros((1, 1, 3, 2), dtype=torch.float64)
    exposure = np.array([[4, 0, 2, 3, 1]])
    result = posterior_field_reconstruction(model, mean, mean.clone(), exposure, unit_chunk_size=2)
    rates = np.array([.5, 1., 2., 3., 4.])
    exact_unit_counts = -np.expm1(-rates * model.tick_seconds) * exposure[0]
    exact_fields = np.array([[exact_unit_counts[1] + exact_unit_counts[3], 0.,
                              exact_unit_counts[0] + exact_unit_counts[2] + exact_unit_counts[4]]])
    np.testing.assert_allclose(result["mean_state_count"], exact_fields, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["posterior_integrated_intensity"], [[4 * .04, 0., 6.5 * .04]],
                               rtol=1e-12, atol=1e-12)
    assert np.all(result["posterior_expected_count_lower"] <= exact_fields + 1e-14)
    assert np.all(exact_fields <= result["posterior_expected_count_upper"] + 1e-14)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", [41, 42, 43])
def test_chunk_invariance_on_cpu_and_available_cuda(device, seed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this test allocation")
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn((1, 5, 3, 2), generator=generator, dtype=torch.float64) * .2
    variance = torch.rand(mean.shape, generator=generator, dtype=torch.float64) * .1
    exposure = torch.randint(0, 5, (5, 5), generator=generator)
    expected = posterior_field_reconstruction(FrozenReadout(), mean, variance, exposure, unit_chunk_size=5)
    model = FrozenReadout().to(device)
    for chunk in (1, 2, 4, 8):
        actual = posterior_field_reconstruction(model, mean, variance, exposure, unit_chunk_size=chunk)
        for name in expected:
            assert isinstance(actual[name], np.ndarray) and actual[name].dtype == np.float64
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("invalid", [
    "negative_exposure", "fractional_exposure", "excess_exposure", "nonfinite_exposure",
    "exposure_shape", "negative_variance", "nonfinite_variance", "posterior_shape", "zero_chunk",
])
def test_bad_posterior_or_native_exposure_rejected(invalid):
    mean, variance, exposure = inputs()
    exposure = exposure.astype(float)
    chunk = 2
    if invalid == "negative_exposure":
        exposure[0, 0] = -1
    elif invalid == "fractional_exposure":
        exposure[0, 0] = .5
    elif invalid == "excess_exposure":
        exposure[0, 0] = 5
    elif invalid == "nonfinite_exposure":
        exposure[0, 0] = np.nan
    elif invalid == "exposure_shape":
        exposure = exposure[:, :-1]
    elif invalid == "negative_variance":
        variance[0, 0, 0, 0] = -.1
    elif invalid == "nonfinite_variance":
        variance[0, 0, 0, 0] = torch.inf
    elif invalid == "posterior_shape":
        mean = mean[0]
    else:
        chunk = 0
    with pytest.raises(ValueError):
        posterior_field_reconstruction(FrozenReadout(), mean, variance, exposure, unit_chunk_size=chunk)


def test_response_covariance_matches_analytic_between_block_formula():
    block = np.array([-1., 0., 1., 2.])
    values = np.empty((4, 3, 3))
    for time in range(3):
        values[:, time, 0] = block + 10 * time
        values[:, time, 1] = 2 * block + 3 * time
        values[:, time, 2] = -3 * block + 2 * time
    left, right = np.array([0, 0]), np.array([1, 2])
    # Across-block variance is 5/3; shared temporal offsets disappear on centering.
    np.testing.assert_allclose(response_covariance(values, left, right, lag=1),
                               [[10 / 3, -5.], [10 / 3, -5.]], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(response_covariance(values, left, right),
                               [[10 / 3, -5.]] * 3, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="two blocks"):
        response_covariance(values[:1], left, right)
    with pytest.raises(ValueError, match="lag"):
        response_covariance(values, left, right, lag=3)
    with pytest.raises(ValueError, match="field pairs"):
        response_covariance(values, left.astype(float), right)
