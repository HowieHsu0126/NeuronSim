"""Source boundary crossing, causal isolation and fixed diagnostic roles."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from Libs.population_dynamics.lfads import LFADSConditionalSimulator, LFADSRandomStreams
from Libs.population_dynamics.lfads_boundary_diagnostic import (
    BRANCHES, PROTOCOL, LFADSBoundaryDiagnostic, _load_protocol, branch_role, main,
)


class BoundaryCrossingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Libs.population_dynamics import lfads
        config = Path(lfads.__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml'
        spec = yaml.safe_load(config.read_text())['lfads_baseline_check']['model']
        # A lag of two exercises encoder dependencies beyond the boundary and
        # beyond the requested future. This is an initialized diagnostic fixture.
        cls.model = LFADSConditionalSimulator([12, 4], 1, 3, 3, .05,
            dict(spec, controller_input_lag=2), initialization_seed=41, device='cpu')
        cls.y = np.array([[[0, 2], [1, 0], [2, 1], [1, 3], [0, 1], [1, 1]]], dtype=np.int64)
        cls.u = np.linspace(0, .8, 6, dtype=np.float32).reshape(1, 6, 1)
        cls.seeds = (410, 411, 412)
        cls.before_weights = cls.model._full.values()
        cls.before_sites = tuple(cls.model._full.randomness.sites)
        cls.diag = LFADSBoundaryDiagnostic(cls.model)
        cls.prepared = cls.diag.prepare(cls.y, cls.u, horizon_bins=2,
                                        n_samples=4, sampling_seeds=cls.seeds)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_public_causal_sample_is_bitwise_reproduced(self):
        actual = self.diag.sample_arm(self.prepared, BRANCHES[0], requested_role='forecast')
        reference = self.model.simulate(self.prepared.prefix_state, self.u[:, 3:5],
                                       LFADSRandomStreams.from_seeds(*self.seeds), 4)
        for key in ('counts', 'conditional_expected_counts', 'initial_latent_states',
                    'historical_controller_inputs', 'future_controller_inputs',
                    'boundary_generator_states', 'future_generator_states'):
            np.testing.assert_array_equal(getattr(actual, key), getattr(reference, key))
        receipt = self.diag.verify_endpoint(self.prepared, BRANCHES[0], actual,
                                            rtol=PROTOCOL['endpoint_rtol'], atol=PROTOCOL['endpoint_atol'])
        self.assertTrue(receipt['sampled_counts_bitwise_equal'])
        self.assertTrue(all(row['bitwise_equal'] for row in receipt['arrays'].values()))

    def test_full_response_endpoint_matches_raw_native_source_with_same_noises(self):
        actual = self.diag.sample_arm(self.prepared, BRANCHES[3])
        source = self.model._full.model
        native_feed = source.build_feed_dict('recording', np.repeat(self.y, 4, axis=0),
                                              np.repeat(self.u, 4, axis=0), keep_prob=1.)
        native_feed[source.posterior_zs_g0.noise_bxn] = self.prepared.initial_standard_noise[:, 0]
        for tick in range(5):
            native_feed[source.posterior_zs_co[tick].noise_bxn] = (
                self.prepared.history_standard_noise[:, tick] if tick < 3
                else self.prepared.future_standard_noise[:, tick - 3])
        native = self.model._read_future(native_feed, 4, 2)
        for key in ('conditional_expected_counts', 'boundary_generator_states', 'future_generator_states'):
            np.testing.assert_array_equal(getattr(actual, key), native[key])
        receipt = self.diag.verify_endpoint(self.prepared, BRANCHES[3], actual,
            rtol=PROTOCOL['endpoint_rtol'], atol=PROTOCOL['endpoint_atol'])
        self.assertTrue(receipt['all_within_tolerance'])

    def test_crossing_preserves_boundary_and_fixed_posterior_donor(self):
        samples = [self.diag.sample_arm(self.prepared, branch) for branch in BRANCHES]
        for key in ('initial_latent_states', 'historical_controller_inputs', 'boundary_generator_states'):
            np.testing.assert_array_equal(getattr(samples[0], key), getattr(samples[2], key))
            np.testing.assert_array_equal(getattr(samples[1], key), getattr(samples[3], key))
        np.testing.assert_array_equal(samples[2].future_controller_inputs, samples[3].future_controller_inputs)
        self.assertFalse(np.array_equal(samples[0].boundary_generator_states, samples[1].boundary_generator_states))
        self.assertFalse(np.array_equal(samples[2].conditional_expected_counts, samples[3].conditional_expected_counts))
        source = self.model._full.model
        phi, logvar = self.model._full.session.run([source.prior_zs_ar_con.phis_bxu,
            source.prior_zs_ar_con.logevars_1xu], feed_dict=self.prepared.full_response_feed)
        for name, sample in zip(('prefix', 'full_response'), samples[:2]):
            previous = self.prepared.history[name][:, 0, -1]
            for tick in range(2):
                previous = (phi * previous + np.exp(.5 * logvar)
                            * self.prepared.future_standard_noise[:, tick]).astype(np.float32)
                np.testing.assert_array_equal(sample.future_controller_inputs[:, 0, tick], previous)

    def test_future_responses_change_diagnostics_but_never_causal_forecast(self):
        changed = self.y.copy()
        changed[:, 3:] += np.array([[[7, 2], [3, 11], [9, 4]]])
        alternate = self.diag.prepare(changed, self.u, horizon_bins=2,
                                     n_samples=4, sampling_seeds=self.seeds)
        for key in ('prefix',):
            np.testing.assert_array_equal(self.prepared.boundary[key], alternate.boundary[key])
        causal = self.diag.sample_arm(self.prepared, BRANCHES[0])
        causal_changed = self.diag.sample_arm(alternate, BRANCHES[0])
        np.testing.assert_array_equal(causal.counts, causal_changed.counts)
        np.testing.assert_array_equal(causal.conditional_expected_counts, causal_changed.conditional_expected_counts)
        self.assertFalse(np.array_equal(self.prepared.boundary['full_response'], alternate.boundary['full_response']))
        self.assertFalse(np.array_equal(self.prepared.posterior_future, alternate.posterior_future))

    def test_complete_lagged_encoders_are_present_and_no_response_buffer_is_retained(self):
        source = self.model._full.model
        self.assertNotIn(source.dataset_ph, self.prepared.full_response_native_feed)
        self.assertNotIn(source.dataset_ph, self.prepared.prefix_feed)
        for tensor in source.ci_enc_fwd + source.ci_enc_rev:
            self.assertIn(tensor, self.prepared.full_response_native_feed)
        changed = dict(self.prepared.full_response_native_feed)
        changed[source.ci_enc_rev[4]] = np.zeros_like(changed[source.ci_enc_rev[4]])
        source_boundary = self.model._full.session.run(source.gen_states[2],
            feed_dict=self.prepared.full_response_native_feed)
        truncated_boundary = self.model._full.session.run(source.gen_states[2], feed_dict=changed)
        self.assertFalse(np.array_equal(source_boundary, truncated_boundary))

    def test_diagnostic_arms_cannot_be_promoted_to_forecasts(self):
        for branch in BRANCHES[1:]:
            with self.assertRaisesRegex(ValueError, 'NONCAUSAL_BRANCH'):
                self.diag.sample_arm(self.prepared, branch, requested_role='forecast')
            self.assertFalse(branch_role(branch)['inverse_intervention_evidence'])
            self.assertFalse(branch_role(branch)['mechanistic_microstate_oracle'])
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_LFADS'):
            branch_role('fullposterior_true_microstate')

    def test_observation_draws_are_separate_from_latent_paths(self):
        alternate = replace(self.prepared, sampling_seeds=(410, 411, 413))
        first = self.diag.sample_arm(self.prepared, BRANCHES[3])
        second = self.diag.sample_arm(alternate, BRANCHES[3])
        np.testing.assert_array_equal(first.conditional_expected_counts, second.conditional_expected_counts)
        self.assertFalse(np.array_equal(first.counts, second.counts))

    def test_boundary_corruption_fails_endpoint_instead_of_relabeling_success(self):
        boundary = dict(self.prepared.boundary)
        boundary['full_response'] = boundary['full_response'] + .25
        altered = replace(self.prepared, boundary=boundary)
        sample = self.diag.sample_arm(altered, BRANCHES[3])
        with self.assertRaisesRegex(ValueError, 'ENDPOINT_MISMATCH'):
            self.diag.verify_endpoint(altered, BRANCHES[3], sample,
                rtol=PROTOCOL['endpoint_rtol'], atol=PROTOCOL['endpoint_atol'])

    def test_no_training_or_source_random_site_mutations(self):
        self.assertEqual(self.model.training_updates, 0)
        self.assertEqual(tuple(self.model._full.randomness.sites), self.before_sites)
        for name, value in self.model._full.values().items():
            np.testing.assert_array_equal(value, self.before_weights[name])

    def test_incomplete_response_and_changed_graph_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ONE_COMPLETE'):
            self.diag.prepare(self.y[:, :5], self.u, horizon_bins=2,
                              n_samples=4, sampling_seeds=self.seeds)
        with mock.patch.object(self.model, 'training_horizon_bins', 4):
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_LFADS'):
                LFADSBoundaryDiagnostic(self.model)


class ProtocolAndCLI(unittest.TestCase):
    def test_frozen_protocol_is_checked_before_any_cohort_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(PROTOCOL))
            self.assertEqual(_load_protocol(config), PROTOCOL)
            changed = dict(PROTOCOL, training_seeds=[41], formal_evaluation=True)
            config.write_text(yaml.safe_dump(changed))
            output = root / 'out'
            argv = ['diagnostic', '--acquisition', str(root / 'missing_data'),
                    '--fields', str(root / 'missing_fields'), '--training', str(root / 'missing_training'),
                    '--config', str(config), '--output', str(output), '--device', 'cpu']
            with mock.patch('sys.argv', argv):
                with self.assertRaisesRegex(ValueError, 'EXACT_FROZEN'):
                    main()
            failure = json.loads((output / 'failure.json').read_text())
            self.assertEqual(failure['status'], 'FAILED')
            self.assertEqual(failure['error_type'], 'ValueError')
            self.assertFalse((output / 'scores.csv').exists())


if __name__ == '__main__':
    unittest.main()
