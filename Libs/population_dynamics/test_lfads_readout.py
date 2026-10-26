"""Pinned native readout oracles; these fixtures do not qualify neural fidelity."""
from pathlib import Path
import inspect

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import yaml

from Libs.external.lfads.utils import init_linear
from .lfads import LFADSConditionalSimulator, LFADSRandomStreams
from .lfads_readout import LFADSReadout, L2_EPSILON

# Numerical reconstruction bounds are fixed before examining real forecasts.
LOGIT_ATOL, LOGIT_RTOL = 1e-5, 1e-6
COUNT_ATOL, COUNT_RTOL = 1e-7, 1e-4


def native_readout(states, readout):
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        g, w, b = readout.generator_weight, readout.output_weight, readout.output_bias
        with tf.variable_scope('gen'):
            projection, _ = init_linear(*g.shape, do_bias=False, mat_init_value=g,
                                        normalized=True, name='gen_2_fac')
        with tf.variable_scope('glm'):
            output, bias = init_linear(*w.shape, mat_init_value=w, bias_init_value=b,
                                       normalized=False, name='fac_2_logrates_recording')
        logits = tf.matmul(tf.matmul(tf.constant(states), projection), output) + bias
        expected = tf.exp(tf.clip_by_value(logits, -readout.clip_value, readout.clip_value))
        initializer = tf.global_variables_initializer()
    with tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU': 0},
                    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as session:
        session.run(initializer)
        return session.run([logits, expected])


def fixture_readout(tiny_column=False):
    g = np.array([[3., -1.], [4., 2.], [0., 2.]], dtype=np.float32)
    if tiny_column: g[:, 1] = [0., 1e-8, -1e-8]
    w = np.linspace(-.7, .8, 14, dtype=np.float32).reshape(2, 7)
    b = np.linspace(-2., 1., 7, dtype=np.float32)[None]
    return LFADSReadout(g, w, b, np.arange(7), .005, 80.)


@pytest.mark.parametrize('tiny_column', [False, True])
@pytest.mark.parametrize('chunk', [1, 3, 7])
def test_readout_matches_native_column_normalization_and_neuron_chunks(tiny_column, chunk):
    assert inspect.signature(tf.nn.l2_normalize).parameters['epsilon'].default == L2_EPSILON
    readout = fixture_readout(tiny_column)
    states = np.array([[0, 0, 0], [1, -2, 3], [-5, 5, -5], [5, -5, 5]], dtype=np.float32)
    logits, truth = native_readout(states, readout)
    actual_logits = readout.log_counts(readout.factors(states), 0, 7)
    np.testing.assert_allclose(actual_logits, logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    mapping = np.array([2, 0, 1, 2, 1, 0, 3])
    fields, _ = readout.expected_field_counts(states, mapping, 4, chunk_units=chunk)
    reference = np.stack([truth[:, mapping == field].sum(axis=1, dtype=np.float64) for field in range(4)], axis=1)
    np.testing.assert_allclose(fields, reference, atol=COUNT_ATOL, rtol=COUNT_RTOL)


def test_saved_native_generator_states_reproduce_full_unit_means(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml').read_text())['lfads_baseline_check']['model']
    with LFADSConditionalSimulator(list(range(7)), 1, 3, 3, .005, config, initialization_seed=41) as model:
        state = model.infer_state({'counts': np.zeros((1, 3, 7)), 'stimulus': np.zeros((1, 3, 1))})
        sample = model.simulate(state, np.array([[[0.], [.2], [.7]]], dtype=np.float32),
                                LFADSRandomStreams.from_seeds(601, 602, 603), 3)
        model.save(tmp_path / 'checkpoint')
    readout = LFADSReadout.from_checkpoint(tmp_path / 'checkpoint')
    mapping = np.arange(7)
    expected, _ = readout.expected_field_counts(sample.future_generator_states, mapping, 7, chunk_units=3)
    np.testing.assert_allclose(expected, sample.conditional_expected_counts, atol=COUNT_ATOL, rtol=COUNT_RTOL)
    assert readout.bin_seconds == sample.bin_seconds and np.array_equal(readout.unit_ids, sample.unit_ids)


@pytest.mark.parametrize('bias', [-80., 80., float('nan')])
def test_readout_preserves_source_clip_and_nonfinite_refusal(bias):
    readout = fixture_readout()
    readout.output_bias[0, 0] = bias
    with pytest.raises(FloatingPointError, match='CLIP_ACTIVE_OR_NONFINITE'):
        readout.expected_field_counts(np.zeros((2, 3), dtype=np.float32), np.arange(7), 7)


@pytest.mark.parametrize('change', ['state_dtype', 'state_shape', 'map_length', 'map_index', 'chunk'])
def test_readout_rejects_wrong_state_or_observation_identity(change):
    readout = fixture_readout()
    states, mapping, chunk = np.zeros((2, 3), dtype=np.float32), np.arange(7), 3
    if change == 'state_dtype': states = states.astype(np.float64)
    elif change == 'state_shape': states = states[:, :2]
    elif change == 'map_length': mapping = mapping[:-1]
    elif change == 'map_index': mapping[0] = 7
    else: chunk = 0
    with pytest.raises(ValueError): readout.expected_field_counts(states, mapping, 7, chunk_units=chunk)
