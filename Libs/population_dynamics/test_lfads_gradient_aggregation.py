"""Numerical cross-policy and exact within-policy LFADS gradient checks."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import yaml

from .lfads import LFADSConditionalSimulator
from .test_lfads import CONFIG, streams

# Declared before numerical comparisons; source gradients and Adam are float32.
CROSS_POLICY_RTOL = 2e-5
CROSS_POLICY_ATOL = 2e-6


def spec():
    return {**yaml.safe_load(CONFIG.read_text())['lfads_baseline_check']['model'],
            'keep_prob': .65, 'controller_input_lag': 2, 'implicit_identity_readin': True}


def check_nested(left, right, *, exact):
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], dict):
            check_nested(left[key], right[key], exact=exact)
        elif exact:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        else:
            np.testing.assert_allclose(left[key], right[key], rtol=CROSS_POLICY_RTOL,
                                       atol=CROSS_POLICY_ATOL, err_msg=key)


def probe(model, counts, stimulus):
    view, source = model._full, model._full.model
    if not hasattr(view, '_aggregation_test_gradients'):
        with view.graph.as_default():
            variables = tf.trainable_variables()
            method = tf.AggregationMethod.EXPERIMENTAL_TREE if source.gradient_aggregation == 'tree' else tf.AggregationMethod.ADD_N
            gradients = tf.gradients(source.cost, variables, aggregation_method=method, name='comparison_gradients')
            assert all(value is not None for value in gradients)
            view._aggregation_test_gradients = dict(zip([v.name for v in variables], gradients))
    feed = source.build_feed_dict('recording', counts.astype(np.int32), stimulus, keep_prob=.65)
    feed[view.randomness.update] = model.training_updates
    return view.session.run(dict(cost=source.cost, reconstruction=source.recon_cost, kl=source.kl_cost,
        norm=source.grad_global_norm, raw_gradients=view._aggregation_test_gradients,
        clipped_gradients=dict(zip([v.name for v in source.train_vars], source.grads))), feed_dict=feed)


def forward(model, counts, stimulus):
    view, source = model._full, model._full.model
    feed = source.build_feed_dict('recording', counts.astype(np.int32), stimulus, keep_prob=.65)
    feed[view.randomness.update] = model.training_updates
    return view.session.run(dict(initial_mean=source.posterior_zs_g0.mean,
        initial_logvar=source.posterior_zs_g0.logvar,
        posterior_samples=[value.sample for value in source.posterior_zs_co],
        states=source.gen_states, outputs=source.output_dist_params, cost=source.cost), feed_dict=feed)


def test_shared_forward_equal_and_gradients_adam_numerically_agree():
    counts = (np.arange(2*6*5).reshape(2,6,5) % 4).astype(np.int64)
    stimulus = np.linspace(0, .8, 12, dtype=np.float32).reshape(2,6,1)
    with LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005, spec(), initialization_seed=41) as default, \
         LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005, {**spec(), 'gradient_aggregation': 'tree'}, initialization_seed=41) as tree:
        assert default._full.model.gradient_aggregation == 'add_n'
        assert tree._full.model.gradient_aggregation == 'tree'
        assert default._full.randomness.sites == tree._full.randomness.sites
        # Aggregation is built after all forward variable initializers.
        check_nested(default._full.values(), tree._full.values(), exact=True)
        tree._full.restore(default._full.values())
        tree._prefix_dirty = True
        check_nested(forward(default, counts, stimulus), forward(tree, counts, stimulus), exact=True)
        for _ in range(3):
            check_nested(probe(default, counts, stimulus), probe(tree, counts, stimulus), exact=False)
            check_nested(default.fit_step(counts, stimulus), tree.fit_step(counts, stimulus), exact=False)
            check_nested(default._full.values(), tree._full.values(), exact=False)


def test_tree_checkpoint_next_updates_and_sample_replay_exact(tmp_path):
    counts = (np.arange(2*6*5).reshape(2,6,5) % 4).astype(np.int64)
    stimulus = np.linspace(0, .8, 12, dtype=np.float32).reshape(2,6,1)
    with LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005, {**spec(), 'gradient_aggregation': 'tree'}, initialization_seed=41) as model:
        model.fit_step(counts, stimulus)
        checkpoint = tmp_path / 'tree_checkpoint'
        model.save(checkpoint)
        record = json.loads((checkpoint/'model.json').read_text())
        assert record['model_config']['gradient_aggregation'] == 'tree'
        with LFADSConditionalSimulator.load(checkpoint) as restored:
            assert not restored.loaded_for_inference
            assert restored._full.model.gradient_aggregation == 'tree'
            assert restored._full.randomness.sites == model._full.randomness.sites
            check_nested(model._full.values(), restored._full.values(), exact=True)
            history = dict(counts=counts[:,:3], stimulus=stimulus[:,:3])
            first = model.simulate(model.infer_state(history), stimulus[:,3:], streams(), 4)
            second = restored.simulate(restored.infer_state(history), stimulus[:,3:], streams(), 4)
            for name in ('counts','conditional_expected_counts','historical_controller_inputs','future_controller_inputs',
                         'initial_latent_states','boundary_generator_states','future_generator_states'):
                np.testing.assert_array_equal(getattr(first,name),getattr(second,name))
            for update in (1,2,3):
                assert model.training_updates == restored.training_updates == update
                check_nested(probe(model, counts, stimulus), probe(restored, counts, stimulus), exact=True)
                check_nested(model.fit_step(counts, stimulus), restored.fit_step(counts, stimulus), exact=True)
                check_nested(model._full.values(), restored._full.values(), exact=True)


def gradient_metadata(model, shape):
    ops = [op for op in model._full.graph.get_operations() if op.type == 'AddN'
           and 'gradients' in op.name and op.outputs[0].shape.as_list() == list(shape)]
    @lru_cache(None)
    def leaves(name):
        op = model._full.graph.get_operation_by_name(name)
        if op.type == 'AddN' and 'gradient_sum' in op.name and op.outputs[0].shape.as_list() == list(shape):
            return tuple(value for tensor in op.inputs for value in leaves(tensor.op.name))
        return (op.name,)
    chains = []
    for op in ops:
        if 'gradient_sum' not in op.name:
            continue
        producers = leaves(op.name)
        if len(producers) >= 16:
            chains.append(dict(sum_op=op.name, partial_producers=list(producers), partial_count=len(producers)))
    # One maximal chain for each encoder direction/weight group.
    groups = {}
    for row in chains:
        key = row['sum_op'].rsplit('/AddN',1)[0]
        if key not in groups or row['partial_count'] > groups[key]['partial_count']:
            groups[key] = row
    for row in groups.values():
        matches = [re.search(r'/(ic|ci)_enc_(fwd|rev)_(\d+)/GRU/Gates/', name)
                   for name in row['partial_producers']]
        assert all(match is not None for match in matches)
        family, direction = matches[0].group(1, 2)
        assert all(match.group(1, 2) == (family, direction) for match in matches)
        steps = [int(match.group(3)) for match in matches]
        encoded_bins = (model._full.model.hps.num_steps_for_gen_ic if family == 'ic'
                        else model._full.model.hps.num_steps)
        row.update(encoder=family, direction=direction, processing_steps=steps,
                   physical_time_indices=[step - 1 if direction == 'fwd' else encoded_bins - step
                                          for step in steps])
    return dict(matrix_shape=list(shape), addn_nodes=len(ops),
                maximum_fanin=max(len(op.inputs) for op in ops),
                representative_chains=list(groups.values()))


def test_unrolled_large_matrix_sums_have_binary_fanin_and_record_order(tmp_path):
    metadata = {}
    config = spec()
    shape = (33 + config['ci_enc_dim'], 2 * config['ci_enc_dim'])
    for policy in ('add_n','tree'):
        with LFADSConditionalSimulator(list(range(33)), 1, 16, 16, .005,
                {**config, 'gradient_aggregation': policy}, initialization_seed=41) as model:
            metadata[policy] = gradient_metadata(model, shape)
    assert metadata['add_n']['maximum_fanin'] >= 16
    assert metadata['tree']['maximum_fanin'] == 2
    # The controller lag excludes the last two processing steps from the loss.
    # Initial-condition encoders use only the sixteen history bins.
    expected_counts = {'ic': 16, 'ci': 32 - config['controller_input_lag']}
    chains = metadata['tree']['representative_chains']
    assert {(row['encoder'], row['direction']) for row in chains} == {
        ('ic', 'fwd'), ('ic', 'rev'), ('ci', 'fwd'), ('ci', 'rev')}
    for row in chains:
        expected = expected_counts[row['encoder']]
        assert row['partial_count'] == expected
        # The left fold consumes the latest used processing step first, matching
        # recurrent backprop's dependency direction. Scheduling still controls RSS.
        assert row['processing_steps'] == list(range(expected, 0, -1))
    (tmp_path/'graph_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


@pytest.mark.parametrize('value',[None, True, 0, 'accumulate_n', 'TREE'])
def test_invalid_gradient_aggregation_fails_precisely(value):
    with pytest.raises(ValueError, match='GRADIENT_AGGREGATION_MUST_BE_ADD_N_OR_TREE'):
        LFADSConditionalSimulator(list(range(5)), 1, 3, 3, .005,
            {**spec(), 'gradient_aggregation': value}, initialization_seed=41)
