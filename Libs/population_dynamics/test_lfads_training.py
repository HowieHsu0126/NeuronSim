"""Engineering evidence for LFADS dropout/posterior RNG and exact training resume.

These bounded fixtures test computational replay, not fitted scientific fidelity.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import yaml

from .lfads import LFADSConditionalSimulator, TRAINING_RANDOM_SCHEMA, _TrainingRandomness
from .test_lfads import CONFIG, streams


def reachable_nodes(graph, fetches):
    """Walk actual data/control dependencies, including invoked graph functions.

    Initializers and prior samplers count only if a training fetch can reach them.
    Function outputs and mandatory control returns seed each invoked body walk.
    """
    definition = graph.as_graph_def()
    functions = {function.signature.name: function for function in definition.library.function}
    bodies = {'': {node.name: node for node in definition.node}}
    bodies.update({name: {node.name: node for node in function.node_def}
                   for name, function in functions.items()})
    pending = [('', (fetch if isinstance(fetch, tf.Operation) else fetch.op).name)
               for fetch in fetches]
    reached = {}
    invoked = set()
    while pending:
        scope, reference = pending.pop()
        name = reference.lstrip('^').split(':', 1)[0]
        identity = (scope, name)
        if identity in reached or name not in bodies[scope]:
            continue
        node = bodies[scope][name]
        reached[identity] = node
        pending.extend((scope, parent) for parent in node.input)
        calls = [node.op] if node.op in functions else []
        for attribute in node.attr.values():
            if attribute.HasField('func'):
                calls.append(attribute.func.name)
            calls.extend(function.name for function in attribute.list.func)
        for called in calls:
            if called in invoked:
                continue
            if called not in functions:
                raise AssertionError('Missing called FunctionDef: ' + called)
            invoked.add(called)
            function = functions[called]
            pending.extend((called, output) for output in function.ret.values())
            pending.extend((called, output) for output in function.control_ret.values())
    return reached


def is_stateful_random(kind):
    if 'Stateless' in kind:
        return False
    return ('Random' in kind or 'Rng' in kind or kind in
            {'Multinomial', 'TruncatedNormal', 'ParameterizedTruncatedNormal',
             'StatefulStandardNormal', 'StatefulStandardNormalV2',
             'StatefulUniform', 'StatefulUniformInt', 'StatefulTruncatedNormal'})


def raw_source_probe(model, counts, stimulus):
    """Read unmodified source loss and unclipped derivatives without an update."""
    view, source = model._full, model._full.model
    if not hasattr(view, '_test_raw_gradients'):
        with view.graph.as_default():
            variables = tf.trainable_variables()
            gradients = tf.gradients(source.cost, variables)
            if any(gradient is None for gradient in gradients):
                raise AssertionError('A native trainable variable has no cost derivative')
            view._test_raw_gradients = {variable.name: gradient
                                        for variable, gradient in zip(variables, gradients)}
    feed = source.build_feed_dict('recording', counts.astype(np.int32), stimulus,
                                  keep_prob=model.model_config['keep_prob'])
    feed[view.randomness.update] = model.training_updates
    fetches = {'source_cost': source.cost, 'reconstruction_nll': source.recon_cost,
               'kl_cost': source.kl_cost, 'l2_cost': source.l2_cost,
               'gradient_norm': source.grad_global_norm,
               'gradients': view._test_raw_gradients}
    return view.session.run(fetches, feed_dict=feed)


class LFADSTrainingRandomnessTests(unittest.TestCase):
    def test_site_update_and_seed_keys_dropout_mask_and_scaling(self):
        graph = tf.Graph()
        with graph.as_default():
            randomness = _TrainingRandomness(41)
            normal = randomness.normal([256])
            other_site = randomness.normal([256])
            masked = randomness.dropout(tf.ones([2048], tf.float32), .5)
            identity = randomness.dropout(tf.ones([32], tf.float32), 1.)
            other_seed = _TrainingRandomness(42)
            seeded = other_seed.normal([256])
        config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2,
                                device_count={'GPU': 0})
        with tf.Session(graph=graph, config=config) as session:
            fetch = [normal, other_site, masked, identity, seeded]
            feed = {randomness.update: 7, other_seed.update: 7}
            first = session.run(fetch, feed_dict=feed)
            repeated = session.run(fetch, feed_dict=feed)
            for left, right in zip(first, repeated):
                np.testing.assert_array_equal(left, right)
            changed = session.run(fetch, feed_dict={randomness.update: 8, other_seed.update: 8})
            for index in (0, 1, 2, 4):
                self.assertFalse(np.array_equal(first[index], changed[index]))
            self.assertFalse(np.array_equal(first[0], first[1]))
            self.assertFalse(np.array_equal(first[0], first[4]))
            self.assertEqual(set(np.unique(first[2])), {0., 2.})
            np.testing.assert_array_equal(first[3], np.ones(32))
        self.assertEqual(randomness.sites, ['normal', 'normal', 'dropout', 'dropout'])

    def test_dependency_walk_sees_called_function_randomness_but_ignores_unused_draws(self):
        graph = tf.Graph()
        with graph.as_default():
            @tf.function(autograph=False)
            def called_randomness(value):
                return value + tf.random_normal([4], name='reachable_function_draw')

            output = called_randomness(tf.ones([4]))
            unused = tf.random_normal([4], name='unused_initializer_draw')
        reached = reachable_nodes(graph, [output])
        offending = [(scope, name, node.op) for (scope, name), node in reached.items()
                     if is_stateful_random(node.op)]
        self.assertTrue(any(scope and 'reachable_function_draw' in name
                            for scope, name, _ in offending), offending)
        self.assertNotIn(('', unused.op.name), reached)


class LFADSTrainingReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = {**yaml.safe_load(CONFIG.read_text())['lfads_baseline_check']['model'],
                    'controller_input_lag': 2, 'keep_prob': .65}
        cls.y = np.array([[[0, 2], [1, 0], [2, 1], [1, 3], [0, 1], [1, 1]],
                          [[2, 1], [1, 0], [0, 1], [0, 2], [1, 0], [0, 1]]], dtype=np.int64)
        cls.u = np.linspace(0, .8, 12, dtype=np.float32).reshape(2, 6, 1)

    def setUp(self):
        self.model = LFADSConditionalSimulator([12, 4], 1, 3, 3, .05, self.spec, initialization_seed=41)

    def tearDown(self):
        self.model.close()

    def assert_nested_exact(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            with self.subTest(value=name):
                if isinstance(left[name], dict):
                    self.assert_nested_exact(left[name], right[name])
                else:
                    np.testing.assert_array_equal(left[name], right[name])

    def test_dropout_training_checkpoint_replays_three_next_updates_and_all_adam_state(self):
        model = self.model
        model.fit_step(self.y, self.u)
        model.fit_step(self.y, self.u)
        self.assertEqual(model.training_updates, 2)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            model.save(checkpoint)
            record = json.loads((checkpoint / 'model.json').read_text())
            self.assertEqual(record['schema'], 'neural_population_lfads_conditional_v2')
            self.assertEqual(record['training_rng_schema'], TRAINING_RANDOM_SCHEMA)
            self.assertEqual(record['training_random_sites'], model._full.randomness.sites)
            self.assertEqual(record['training_updates'], 2)
            self.assertEqual(record['training_history_bins'], 3)
            self.assertEqual(record['training_horizon_bins'], 3)
            self.assertTrue(record['training_rng_resume_supported'])
            with LFADSConditionalSimulator.load(checkpoint) as resumed:
                self.assertFalse(resumed.loaded_for_inference)
                self.assert_nested_exact(model._full.values(), resumed._full.values())
                self.assertTrue(any('Adam' in name for name in model._full.variables))
                for update in (2, 3, 4):
                    self.assertEqual(model.training_updates, update)
                    self.assertEqual(resumed.training_updates, update)
                    self.assert_nested_exact(raw_source_probe(model, self.y, self.u),
                                             raw_source_probe(resumed, self.y, self.u))
                    self.assert_nested_exact(model.fit_step(self.y, self.u),
                                             resumed.fit_step(self.y, self.u))
                    self.assert_nested_exact(model._full.values(), resumed._full.values())
                self.assertEqual(model.training_updates, 5)
                self.assertEqual(resumed.training_updates, 5)

    def test_inference_and_explicit_sample_streams_cannot_advance_training_randomness(self):
        model = self.model
        model.fit_step(self.y, self.u)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            model.save(checkpoint)
            with LFADSConditionalSimulator.load(checkpoint) as uninterrupted:
                before = model._full.values()
                state = model.infer_state({'counts': self.y[:, :3], 'stimulus': self.u[:, :3]})
                model.simulate(state, self.u[:, 3:], streams(), 3)
                model.simulate(state, self.u[:, 3:], streams(initial=61, dynamics=62, observation=63), 2)
                self.assertEqual(model.training_updates, 1)
                self.assert_nested_exact(before, model._full.values())
                self.assert_nested_exact(raw_source_probe(model, self.y, self.u),
                                         raw_source_probe(uninterrupted, self.y, self.u))
                self.assert_nested_exact(model.fit_step(self.y, self.u),
                                         uninterrupted.fit_step(self.y, self.u))
                self.assert_nested_exact(model._full.values(), uninterrupted._full.values())

    def test_native_gaussian_sample_and_log_probability_share_each_single_noise(self):
        view, source = self.model._full, self.model._full.model
        posteriors = [source.posterior_zs_g0, *source.posterior_zs_co]
        with view.graph.as_default():
            fetches = [{'mean': posterior.mean, 'logvar': posterior.logvar,
                        'noise': posterior.noise_bxn, 'sample': posterior.sample,
                        'logp': posterior.logp()} for posterior in posteriors]
        feed = source.build_feed_dict('recording', self.y.astype(np.int32), self.u, keep_prob=.65)
        feed[view.randomness.update] = 3
        first = view.session.run(fetches, feed_dict=feed)
        repeated = view.session.run(fetches, feed_dict=feed)
        for values, again in zip(first, repeated):
            self.assert_nested_exact(values, again)
            mean, logvar, noise = (values[name].astype(np.float64) for name in ('mean', 'logvar', 'noise'))
            np.testing.assert_allclose(values['sample'], mean + np.exp(.5 * logvar) * noise,
                                       atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(values['logp'], -.5 * (logvar + np.log(2 * np.pi) + noise ** 2),
                                       atol=2e-7, rtol=2e-6)
        feed[view.randomness.update] = 4
        changed = view.session.run(fetches, feed_dict=feed)
        for values, other in zip(first, changed):
            self.assertFalse(np.array_equal(values['noise'], other['noise']))

    def test_training_fetches_have_no_reachable_stateful_random_ops(self):
        model, source = self.model, self.model._full.model
        raw_source_probe(model, self.y, self.u)
        fetches = [source.cost, source.recon_cost, source.kl_cost, source.l2_cost,
                   source.kl_weight, source.l2_weight, source.grad_global_norm, source.train_op,
                   *model._full._test_raw_gradients.values()]
        reached = reachable_nodes(model._full.graph, fetches)
        offending = [(scope, name, node.op) for (scope, name), node in reached.items()
                     if is_stateful_random(node.op)]
        self.assertEqual(offending, [])
        self.assertTrue(any('Stateless' in node.op and 'Normal' in node.op for node in reached.values()))
        self.assertTrue(any('Stateless' in node.op and 'Uniform' in node.op for node in reached.values()))
        # Initializers and generative-prior draws remain the original source ops,
        # but must not be ancestors of the actual posterior-training fetches.
        unused = [operation.name for operation in model._full.graph.get_operations()
                  if is_stateful_random(operation.type) and ('', operation.name) not in reached]
        self.assertTrue(unused)
        total_bins = model.history_bins + model.horizon_bins
        self.assertEqual(model._full.randomness.sites.count('normal'), 1 + total_bins)
        self.assertEqual(model._full.randomness.sites.count('dropout'), 3 + 2 * total_bins)

    def test_invalid_keep_probability_and_checkpoint_counter_are_rejected(self):
        for invalid in (0., -1., 1.01, float('nan'), float('inf'), True, None):
            with self.subTest(keep_prob=invalid), self.assertRaisesRegex(ValueError, 'KEEP_PROBABILITY'):
                LFADSConditionalSimulator([12, 4], 1, 3, 3, .05,
                    {**self.spec, 'keep_prob': invalid}, initialization_seed=41)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            self.model.save(checkpoint)
            original = json.loads((checkpoint / 'model.json').read_text())
            for invalid in (-1, True, 1.5, 2**63, None):
                record = {**original, 'training_updates': invalid}
                (checkpoint / 'model.json').write_text(json.dumps(record))
                with self.subTest(counter=invalid), self.assertRaisesRegex(ValueError, 'TRAINING_RANDOM_STATE_INVALID'):
                    with LFADSConditionalSimulator.load(checkpoint):
                        self.fail('Invalid checkpoint counter was accepted')
            record = {**original, 'training_random_sites': original['training_random_sites'] + ['normal']}
            (checkpoint / 'model.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'TRAINING_RANDOM_SITES_CHANGED'):
                with LFADSConditionalSimulator.load(checkpoint):
                    self.fail('Changed random-site layout was accepted')

    def test_changed_horizon_retains_inference_only_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            self.model.save(checkpoint)
            with LFADSConditionalSimulator.load(checkpoint, horizon_bins=4) as changed:
                self.assertTrue(changed.loaded_for_inference)
                self.assertIsNone(changed.training_updates)
                self.assertEqual(changed.training_horizon_bins, 3)
                self.assertEqual(changed.training_history_bins, 3)
                with self.assertRaisesRegex(ValueError, 'TRAINING_RNG_RESUME'):
                    changed.fit_step(self.y, self.u)


    def test_changed_history_and_equal_total_lengths_are_inference_only(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            self.model.save(checkpoint)
            for options in ({'history_bins': 4}, {'history_bins': 4, 'horizon_bins': 2}):
                with self.subTest(options=options), LFADSConditionalSimulator.load(checkpoint, **options) as changed:
                    self.assertTrue(changed.loaded_for_inference)
                    self.assertIsNone(changed.training_updates)
                    self.assertEqual(changed.history_bins, 4)
                    self.assertEqual(changed.training_history_bins, 3)
                    self.assertEqual(changed.training_horizon_bins, 3)
                    with self.assertRaisesRegex(ValueError, 'TRAINING_RNG_RESUME'):
                        changed.fit_step(self.y, self.u)
            with LFADSConditionalSimulator.load(checkpoint, history_bins=4, horizon_bins=2) as changed:
                inference = Path(folder) / 'inference'
                changed.save(inference)
            saved = json.loads((inference / 'model.json').read_text())
            self.assertEqual((saved['history_bins'], saved['horizon_bins']), (4, 2))
            self.assertEqual((saved['training_history_bins'], saved['training_horizon_bins']), (3, 3))
            self.assertFalse(saved['training_rng_resume_supported'])
            with LFADSConditionalSimulator.load(inference) as restored:
                self.assertEqual((restored.history_bins, restored.horizon_bins), (4, 2))
                self.assertEqual((restored.training_history_bins, restored.training_horizon_bins), (3, 3))
                self.assertTrue(restored.loaded_for_inference)
            # Returning an inference-only artifact to its original lengths does
            # not turn it back into a checkpoint with a resumable RNG state.
            with LFADSConditionalSimulator.load(inference, history_bins=3, horizon_bins=3) as restored:
                with self.assertRaisesRegex(ValueError, 'TRAINING_RNG_RESUME'):
                    restored.fit_step(self.y, self.u)

    def test_old_v2_history_metadata_default_and_explicit_equal_lengths_resume(self):
        self.model.fit_step(self.y, self.u)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            self.model.save(checkpoint)
            record = json.loads((checkpoint / 'model.json').read_text())
            record.pop('training_history_bins')
            (checkpoint / 'model.json').write_text(json.dumps(record))
            with LFADSConditionalSimulator.load(checkpoint) as default, LFADSConditionalSimulator.load(
                    checkpoint, history_bins=3, horizon_bins=3) as explicit:
                for restored in (default, explicit):
                    self.assertEqual(restored.training_history_bins, 3)
                    self.assertEqual(restored.training_horizon_bins, 3)
                    self.assertFalse(restored.loaded_for_inference)
                    self.assert_nested_exact(self.model._full.values(), restored._full.values())
                    self.assert_nested_exact(raw_source_probe(self.model, self.y, self.u),
                                             raw_source_probe(restored, self.y, self.u))
                expected = self.model.fit_step(self.y, self.u)
                for restored in (default, explicit):
                    self.assert_nested_exact(expected, restored.fit_step(self.y, self.u))
                    self.assert_nested_exact(self.model._full.values(), restored._full.values())

    def test_invalid_training_history_metadata_and_history_override_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            self.model.save(checkpoint)
            path = checkpoint / 'model.json'
            original = json.loads(path.read_text())
            for invalid in (0, -1, True, 1.5, None):
                path.write_text(json.dumps({**original, 'training_history_bins': invalid}))
                with self.subTest(training_history=invalid), self.assertRaisesRegex(ValueError, 'TRAINING_LENGTHS_INVALID'):
                    LFADSConditionalSimulator.load(checkpoint)
            path.write_text(json.dumps(original))
            for invalid in (0, -1, True, 1.5):
                with self.subTest(history_override=invalid), self.assertRaisesRegex(ValueError, 'INPUT_HISTORY_OR_HORIZON'):
                    LFADSConditionalSimulator.load(checkpoint, history_bins=invalid)


if __name__ == '__main__':
    unittest.main()
