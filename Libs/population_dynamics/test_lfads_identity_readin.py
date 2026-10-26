"""Algebraic identity-readin checks; no fitted-fidelity or convergence claim."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import yaml

from Libs.external.lfads.lfads import LFADS as SourceLFADS
from .lfads import LFADSConditionalSimulator, _TrainingRandomness
from .test_lfads import CONFIG, streams
from .test_lfads_training import raw_source_probe


def specification():
    return {**yaml.safe_load(CONFIG.read_text())['lfads_baseline_check']['model'],
            'keep_prob': .65, 'controller_input_lag': 2}


def equivalent(left, right):
    assert set(left) == set(right)
    for name in left:
        if isinstance(left[name], dict):
            equivalent(left[name], right[name])
        else:
            np.testing.assert_allclose(left[name], right[name], rtol=2e-6, atol=2e-6, err_msg=name)


def forward_values(model, counts, stimulus):
    native = model._full.model
    feed = native.build_feed_dict('recording', counts.astype(np.int32), stimulus, keep_prob=.65)
    feed[model._full.randomness.update] = model.training_updates
    return model._full.session.run(dict(initial_mean=native.posterior_zs_g0.mean,
        initial_logvar=native.posterior_zs_g0.logvar,
        controller_means=[value.mean for value in native.posterior_zs_co],
        controller_logvars=[value.logvar for value in native.posterior_zs_co],
        generator_states=native.gen_states, output_distribution=native.output_dist_params), feed_dict=feed)


@pytest.mark.parametrize('train_readin', [False, True])
def test_dense_and_implicit_match_forward_loss_gradients_adam_and_sampling(train_readin, tmp_path):
    spec = {**specification(), 'do_train_readin': train_readin}
    counts = (np.arange(2 * 6 * 5).reshape(2, 6, 5) % 4).astype(np.int64)
    stimulus = np.linspace(0, .8, 12, dtype=np.float32).reshape(2, 6, 1)
    with LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005, spec, initialization_seed=41) as dense, \
         LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005,
             {**spec, 'implicit_identity_readin': True}, initialization_seed=41) as implicit:
        assert dense._full.model.implicit_identity_readin is False
        assert implicit._full.model.implicit_identity_readin is True
        assert set(dense._full.variables) == set(implicit._full.variables)
        assert not any('x_2_infac' in name for name in dense._full.variables)
        assert dense._full.randomness.sites == implicit._full.randomness.sites
        # Assignment isolates the algebraic change from TF initializer op seeds.
        implicit._full.restore(dense._full.values())
        implicit._prefix_dirty = True
        for update in range(2):
            assert dense.training_updates == implicit.training_updates == update
            equivalent(forward_values(dense, counts, stimulus), forward_values(implicit, counts, stimulus))
            equivalent(raw_source_probe(dense, counts, stimulus), raw_source_probe(implicit, counts, stimulus))
            equivalent(dense.fit_step(counts, stimulus), implicit.fit_step(counts, stimulus))
            equivalent(dense._full.values(), implicit._full.values())
        assert any('Adam' in name for name in implicit._full.variables)
        history = dict(counts=counts[:, :3], stimulus=stimulus[:, :3])
        first_state, second_state = dense.infer_state(history), implicit.infer_state(history)
        for name in ('initial_mean', 'initial_log_variance', 'controller_encoding_forward', 'controller_encoding_reverse'):
            np.testing.assert_allclose(getattr(first_state, name), getattr(second_state, name), atol=2e-6, rtol=2e-6)
        first = dense.simulate(first_state, stimulus[:, 3:], streams(), 4)
        second = implicit.simulate(second_state, stimulus[:, 3:], streams(), 4)
        assert first.unit_ids == second.unit_ids == tuple(range(5))
        assert first.counts.shape == second.counts.shape == (4, 2, 3, 5)
        np.testing.assert_array_equal(first.counts, second.counts)
        for name in ('conditional_expected_counts', 'initial_latent_states', 'historical_controller_inputs',
                     'future_controller_inputs', 'boundary_generator_states', 'future_generator_states'):
            np.testing.assert_allclose(getattr(first, name), getattr(second, name), atol=2e-6, rtol=2e-6)
        matrix = dense._full.session.run(dense._full.model.fns_in_fac_Ws[0]())
        np.testing.assert_array_equal(matrix, np.eye(5, dtype=np.float32))
        with pytest.raises(NotImplementedError, match='DENSE_MATRIX_EXPORT_UNSUPPORTED'):
            implicit._full.model.fns_in_fac_Ws[0]()
        np.testing.assert_array_equal(implicit._full.session.run(implicit._full.model.fns_in_fatcor_bs[0]()), np.zeros(5))
        checkpoint = tmp_path / 'implicit_checkpoint'
        implicit.save(checkpoint)
        with LFADSConditionalSimulator.load(checkpoint) as resumed:
            assert resumed.model_config['implicit_identity_readin'] is True
            assert resumed.training_updates == implicit.training_updates
            equivalent(implicit._full.values(), resumed._full.values())
            equivalent(raw_source_probe(implicit, counts, stimulus), raw_source_probe(resumed, counts, stimulus))
            replay = resumed.simulate(resumed.infer_state(history), stimulus[:, 3:], streams(), 4)
            np.testing.assert_array_equal(second.counts, replay.counts)


def make_source(spec, folder, datasets):
    names = list(datasets)
    dimensions = {name: data['train_data'].shape[-1] for name, data in datasets.items()}
    hparams = {**spec, 'dataset_names': names, 'dataset_dims': dimensions, 'ndatasets': len(names),
               'batch_size': 1, 'ext_input_dim': 1, 'output_dist': 'poisson', 'num_steps': 6,
               'num_steps_for_gen_ic': 3, 'max_ckpt_to_keep': 1, 'max_ckpt_to_keep_lve': 1,
               'lfads_save_dir': str(folder), 'feedback_factors_or_rates': 'factors'}
    return SourceLFADS(SimpleNamespace(**hparams), kind='train', datasets=datasets,
                       random_ops=_TrainingRandomness(41))


@pytest.mark.parametrize('unsupported', ['multiple_sessions', 'alignment_matrix_cxf', 'alignment_bias_c'])
def test_implicit_mode_rejects_alignment_and_multiple_sessions(unsupported, tmp_path):
    data = {'recording': {'train_data': np.zeros((1, 6, 5), dtype=np.int64)}}
    if unsupported == 'multiple_sessions':
        data['other'] = {'train_data': np.zeros((1, 6, 5), dtype=np.int64)}
    elif unsupported == 'alignment_matrix_cxf':
        data['recording'][unsupported] = np.zeros((5, specification()['factors_dim']), dtype=np.float32)
    else:
        data['recording'][unsupported] = np.zeros(5, dtype=np.float32)
    with tf.Graph().as_default(), tf.device('/cpu:0'), pytest.raises(ValueError, match='ONE_UNALIGNED_SESSION'):
        make_source({**specification(), 'implicit_identity_readin': True}, tmp_path, data)


def test_default_alignment_path_is_unchanged(tmp_path):
    spec = specification()
    alignment = np.arange(5 * spec['factors_dim'], dtype=np.float32).reshape(5, spec['factors_dim']) / 10
    data = {'recording': {'train_data': np.zeros((1, 6, 5), dtype=np.int64), 'alignment_matrix_cxf': alignment}}
    graph = tf.Graph()
    config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2, device_count={'GPU': 0})
    with graph.as_default(), tf.Session(graph=graph, config=config).as_default() as session, tf.device('/cpu:0'):
        source = make_source(spec, tmp_path, data)
        try:
            session.run(tf.global_variables_initializer())
            assert source.implicit_identity_readin is False
            np.testing.assert_array_equal(session.run(source.fns_in_fac_Ws[0]()), alignment)
        finally:
            source.writer.close()


def test_implicit_mode_flag_is_explicit_boolean():
    for invalid in (1, 'true', None):
        with pytest.raises(ValueError, match='FLAG_MUST_BE_BOOLEAN'):
            LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005,
                {**specification(), 'implicit_identity_readin': invalid}, initialization_seed=41)


def test_100000_unit_short_graph_never_builds_dense_identity():
    original_eye = np.eye
    def bounded_eye(n, *args, **kwargs):
        if n == 100000:
            raise AssertionError('quadratic100000-unit identity allocation attempted')
        return original_eye(n, *args, **kwargs)
    with patch('Libs.external.lfads.utils.np.eye', side_effect=bounded_eye), \
         contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
         LFADSConditionalSimulator(list(range(100000)), 1, 2, 1, .005,
             {**specification(), 'implicit_identity_readin': True}, initialization_seed=41) as model:
        assert model._full.model.output_dist_params[0].shape.as_list()[-1] == 100000
        for view in (model._full, model._prefix):
            definition = view.graph.as_graph_def()
            nodes = list(definition.node)
            for function in definition.library.function:
                nodes.extend(function.node_def)
            for node in nodes:
                if node.op == 'Const':
                    shape = tuple(dim.size for dim in node.attr['value'].tensor.tensor_shape.dim)
                    assert shape != (100000, 100000)
            assert definition.ByteSize() < 100 * 1024 * 1024
            assert view.model.implicit_identity_readin is True
            with pytest.raises(NotImplementedError, match='DENSE_MATRIX_EXPORT_UNSUPPORTED'):
                view.model.fns_in_fac_Ws[0]()
