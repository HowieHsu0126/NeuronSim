"""Compare fitted float32 prior parameters with the pinned native LFADS density.

These complete synthetic paths test the numerical interface only; no posterior
cohort is extracted and no trained model or optimizer state is changed.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from Libs.external.lfads.distributions import LearnableAutoRegressive1Prior
from Libs.population_dynamics.lfads_stationary_prior import (
    StationaryARMoments, fit_stationary_ar_prior, stationary_ar_nll,
)


# Float32 source arithmetic is compared to the float64 mathematical density.
RTOL = 2e-5
ATOL = 2e-6


def complete_paths(length):
    rng = np.random.RandomState(96317)  # Fixture randomness, not a training seed.
    data = np.empty((3, length, 4), dtype=np.float32)
    scale = np.array([.3, .7, 1., 2.], dtype=np.float32)
    data[:, 0] = rng.normal(size=(3, 4)).astype(np.float32) * scale
    for tick in range(1, length):
        data[:, tick] = (.7 * data[:, tick-1]
                        + rng.normal(size=(3, 4)).astype(np.float32) * scale)
    return data


@pytest.mark.parametrize('length', [5, 1200])
def test_rounded_fit_matches_native_transforms_and_complete_path_density(length):
    data = complete_paths(length)
    moments = StationaryARMoments(sequence_length=length).update(data)
    fitted = fit_stationary_ar_prior(moments)
    logtau, logvar = fitted['logatau'], fitted['logevars']
    expected = stationary_ar_nll(moments, logtau, logvar)
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        current = tf.placeholder(tf.float32, [None, 4])
        previous = tf.placeholder(tf.float32, [None, 4])
        with tf.variable_scope('LFADS/z'):
            prior = LearnableAutoRegressive1Prior(
                tf.shape(current)[0], 4, tf.ones([4]), tf.ones([4]),
                True, True, 1, 'u_prior_ar1')
        first_density = prior.logp_t(current)
        transition_density = prior.logp_t(current, previous)
        variables = tf.global_variables()
        initialize = tf.global_variables_initializer()
        assign = [tf.assign(prior.logataus_1xu, logtau),
                  tf.assign(prior.logevars_1xu, logvar)]
        transform_fetch = dict(phi=prior.phis_bxu,
            innovation_variance=tf.exp(prior.logevars_1xu),
            stationary_variance=tf.exp(prior.logpvars_1xu),
            innovation_std=tf.exp(.5 * prior.logevars_1xu))
    assert {v.name for v in variables} == {
        'LFADS/z/u_prior_ar1/logevars:0', 'LFADS/z/u_prior_ar1/logatau:0'}
    assert all(v.shape.as_list() == [1, 4] for v in variables)
    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=False,
                           intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=graph, config=config) as session:
        session.run(initialize)
        session.run(assign)
        first, transforms = session.run([first_density, transform_fetch],
                                       {current: data[:, 0]})
        transitions = session.run(transition_density, {
            current: data[:, 1:].reshape(-1, 4),
            previous: data[:, :-1].reshape(-1, 4)})
        saved = session.run([prior.logataus_1xu, prior.logevars_1xu])
    np.testing.assert_array_equal(saved[0], logtau)
    np.testing.assert_array_equal(saved[1], logvar)
    assert first.shape == (3, 4) and transitions.shape == (3*(length-1), 4)
    for name in ('phi', 'innovation_variance', 'stationary_variance'):
        actual = transforms[name][0]
        np.testing.assert_allclose(actual, expected[name], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(transforms['innovation_std'][0],
        np.sqrt(expected['innovation_variance']), rtol=RTOL, atol=ATOL)
    phi = np.array(expected['phi'])
    variance = np.array(expected['innovation_variance'])
    initial_variance = np.array(expected['stationary_variance'])
    direct_first = -.5*(np.log(2*np.pi*initial_variance)
                        + data[:, 0].astype(np.float64)**2/initial_variance)
    residual = (data[:, 1:].astype(np.float64)
                - phi*data[:, :-1].astype(np.float64))
    direct_transitions = -.5*(np.log(2*np.pi*variance) + residual**2/variance)
    np.testing.assert_allclose(first, direct_first, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(transitions, direct_transitions.reshape(-1, 4),
                               rtol=RTOL, atol=ATOL)
    native_nll = -first.astype(np.float64).sum(0)-transitions.astype(np.float64).sum(0)
    np.testing.assert_allclose(native_nll, expected['per_channel'], rtol=RTOL, atol=ATOL)
    # Both density components must contribute; transition-only fitting is different.
    assert np.all(np.abs(first.astype(np.float64).sum(0)) > 1e-3)
