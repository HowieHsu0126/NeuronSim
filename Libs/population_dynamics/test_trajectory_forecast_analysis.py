import itertools

import numpy as np

from Libs.population_dynamics.trajectory_forecast_analysis import covariance_components, replay_scores


def test_covariance_removes_ensemble_mean_noise():
    estimates = []
    for noise in itertools.product((-1., 1.), repeat=4):
        x = np.asarray(noise).reshape(2, 2) + np.array([0., 2.])[:, None]
        sample = np.stack((x, -x), axis=-1)[:, :, None, :]
        observed = np.zeros((2, 1, 2))
        _, covariance, _, _ = covariance_components(sample, observed, np.array([0]), np.array([1]))
        estimates.append(covariance.item())
    # Conditional covariance -1 plus sample covariance of the block means -2.
    assert np.mean(estimates) == -3.


def test_replay_uses_distinct_draw_products():
    sample = np.array([[[0., 2.]], [[2., 0.]], [[0., 4.]]])
    truth = np.array([[1., 1.]])
    result = replay_scores(sample, truth, {"spatial": np.array([[0, 1]])}, [])
    # Pairwise products of variogram draws [2, 2, 4] average to 20/3.
    assert np.isclose(result["variogram_spatial"], 20/3)
    first = np.mean([np.sqrt(2), np.sqrt(2), np.sqrt(10)])
    second = np.mean([np.sqrt(8), 2., np.sqrt(20)])
    assert np.isclose(result["energy_score"], (first-.5*second)/np.sqrt(2))
