"""Synthetic contract and archive checks; no source runtime or model execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

from . import v1_lgn_control_causality as causality
from .provenance import freeze_json, sha256_file

from .v1_lgn_control_causality import (
    compare_rate_prefix, observe_convolutions, validate_config_pair,
    validate_inventory, validate_movie_witness,
)


def baseline_config():
    return dict(scope='development_source_runtime_qualification',
        unit_coverage='all_source_units', converter_python='/usr/bin/python3',
        stimulus=dict(duration_s=3.0, gray_screen_s=0.5,
            spatial_frequency_cycles_per_degree=0.04, temporal_frequency_hz=2.0,
            direction_deg=270.0, contrast_percent=80.0))


class ControlContractTests(unittest.TestCase):
    def setUp(self):
        self.ids = [10, 21, 30]
        self.clock = np.arange(6)/1000.0
        self.rates = np.ones((3, 6))

    def compare(self, control, clock=None):
        return compare_rate_prefix(self.ids, self.clock, self.rates,
            self.clock if clock is None else clock, control, changed_frame=3)

    def test_representative_identity_and_model_mismatch_rejected(self):
        validate_inventory(self.ids, ['a', 'b', 'c'], self.ids, ['a', 'b', 'c'])
        for ids, models in (([10, 22, 30], ['a', 'b', 'c']),
                            ([10, 21, 21], ['a', 'b', 'c']),
                            ([21, 10, 30], ['b', 'a', 'c']),
                            (self.ids, ['a', 'c', 'b'])):
            with self.subTest(ids=ids, models=models):
                with self.assertRaisesRegex(ValueError, 'REPRESENTATIVE_IDENTITY_DIFFERS'):
                    validate_inventory(self.ids, ['a', 'b', 'c'], ids, models)

    def test_exact_prefix_and_no_later_response_are_valid(self):
        result = self.compare(self.rates.copy())
        self.assertTrue(result['unchanged_prefix']['exact'])
        self.assertEqual(result['unchanged_prefix']['mismatch_count'], 0)
        self.assertEqual(result['unchanged_prefix']['affected_source_ids'], [])
        self.assertEqual(result['at_and_after_changed_frame']['changed_source_ids'], [])

    def test_prefix_difference_reports_exact_location_and_size(self):
        changed = self.rates.copy()
        changed[1, 2] += 0.125
        result = self.compare(changed)
        self.assertFalse(result['unchanged_prefix']['exact'])
        self.assertEqual(result['unchanged_prefix']['mismatch_count'], 1)
        self.assertEqual(result['unchanged_prefix']['max_absolute_difference_hz'], 0.125)
        self.assertEqual(result['unchanged_prefix']['affected_source_ids'], [21])
        self.assertEqual(result['at_and_after_changed_frame']['changed_source_ids'], [])

    def test_change_at_first_changed_frame_is_not_a_prefix_failure(self):
        changed = self.rates.copy()
        changed[1, 3:] += 0.25
        result = self.compare(changed)
        self.assertTrue(result['unchanged_prefix']['exact'])
        self.assertEqual(result['at_and_after_changed_frame']['changed_source_ids'], [21])
        self.assertEqual(result['at_and_after_changed_frame']['unchanged_source_ids'], [10, 30])
        self.assertFalse(result['at_and_after_changed_frame']['response_required_for_every_unit'])

    def test_one_representable_clock_difference_is_rejected(self):
        changed_clock = self.clock.copy()
        changed_clock[1] = np.nextafter(changed_clock[1], np.inf)
        with self.assertRaisesRegex(ValueError, 'CLOCK_DIFFERS'):
            self.compare(self.rates, clock=changed_clock)

    def test_only_half_contrast_and_coverage_may_change(self):
        baseline = baseline_config()
        control = deepcopy(baseline)
        control['unit_coverage'] = 'model_representatives'
        control['stimulus']['contrast_percent'] = 40.0
        validate_config_pair(baseline, control)
        for field, value in (('contrast_percent', 20.0), ('gray_screen_s', 0.0),
                             ('direction_deg', 90.0), ('duration_s', 4.0)):
            modified = deepcopy(control)
            modified['stimulus'][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, 'ONLY_HALF_CONTRAST'):
                    validate_config_pair(baseline, modified)
        modified = deepcopy(control)
        modified['unit_coverage'] = 'all_source_units'
        with self.assertRaisesRegex(ValueError, 'ONLY_HALF_CONTRAST'):
            validate_config_pair(baseline, modified)
        modified = deepcopy(control)
        modified['converter_python'] = '/different/python'
        with self.assertRaisesRegex(ValueError, 'ONLY_HALF_CONTRAST'):
            validate_config_pair(baseline, modified)

    def test_movie_recipe_and_prefix_are_bound_to_baseline(self):
        movie = dict(status='SOURCE_MOVIE_CONTROL_WITNESS_PASS', gray_frames=501,
            first_retained_grating_local_frame=1, source_default_phase=0.0,
            frame_rate_hz=1000.0, movie_shape=[3001, 48, 96],
            effective_create_movie_arguments=dict(t_min=0, t_max=3.0, gray_screen_dur=0.5,
                cpd=0.04, temporal_f=2.0, theta=270.0, contrast=0.8),
            checks={name: True for name in ('half_contrast_exact', 'combined_gray_exact',
                'gray_is_zero', 'retained_grating_exact', 'gray_join_frame_count_exact',
                'first_nonzero_frame_follows_gray')})
        validate_movie_witness(movie, baseline_config())
        modified = deepcopy(movie)
        modified['effective_create_movie_arguments']['contrast'] = 0.4
        with self.assertRaisesRegex(ValueError, 'MOVIE_CASE_OR_UNCHANGED_PREFIX'):
            validate_movie_witness(modified, baseline_config())
        modified = deepcopy(movie)
        modified['gray_frames'] = 500
        with self.assertRaisesRegex(ValueError, 'MOVIE_CASE_OR_UNCHANGED_PREFIX'):
            validate_movie_witness(modified, baseline_config())


class ConvolutionObservationTests(unittest.TestCase):
    def test_observation_does_not_force_method_or_change_arrays(self):
        first, second = np.zeros(5), np.ones(3)
        returned = object()
        calls, choices, observations = [], [], []

        def original(in1, in2, mode='full', method='auto'):
            calls.append((in1, in2, mode, method))
            return returned

        def choose(in1, in2, mode='full', measure=False):
            choices.append((in1, in2, mode, measure))
            return 'fft'

        signal = SimpleNamespace(convolve=original, choose_conv_method=choose)
        with observe_convolutions(signal, observations):
            self.assertIs(signal.convolve(first, second, mode='valid'), returned)
        self.assertIs(signal.convolve, original)
        self.assertIs(calls[0][0], first)
        self.assertIs(calls[0][1], second)
        self.assertEqual(calls[0][2:], ('valid', 'auto'))
        self.assertEqual(choices[0][2:], ('valid', False))
        self.assertEqual(observations[0]['input_shapes'], [[5], [3]])
        self.assertEqual(observations[0]['input_dtypes'], ['float64', 'float64'])
        self.assertEqual(observations[0]['requested_method'], 'auto')
        self.assertEqual(observations[0]['chosen_method'], 'fft')
        self.assertTrue(observations[0]['completed'])

    def test_unexpected_convolution_error_is_preserved_and_wrapper_restored(self):
        def original(in1, in2, mode='full', method='auto'):
            raise RuntimeError('source convolution failed')

        def choose(*args, **kwargs):
            self.fail('explicit method must not invoke automatic selection')

        signal = SimpleNamespace(convolve=original, choose_conv_method=choose)
        observations = []
        with self.assertRaisesRegex(RuntimeError, 'source convolution failed'):
            with observe_convolutions(signal, observations):
                signal.convolve(np.ones(3), np.ones(2), 'valid', 'direct')
        self.assertIs(signal.convolve, original)
        self.assertEqual(observations[0]['chosen_method'], 'direct')
        self.assertFalse(observations[0]['completed'])
        self.assertEqual(observations[0]['error_type'], 'RuntimeError')


class FullArchiveComparisonTests(unittest.TestCase):
    def test_full_config_ignores_only_unused_spike_metadata(self):
        baseline = baseline_config()
        control = deepcopy(baseline)
        control['stimulus']['contrast_percent'] = 40.0
        control['spike_generation'] = {'starting_noise_seed': 12345, 'trials': 1}
        causality.validate_full_config_pair(baseline, control)
        for key, value in (('converter_python', '/different/python'), ('extra_rate_setting', True)):
            changed = deepcopy(control)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'MAY_ONLY_HALVE_CONTRAST'):
                causality.validate_full_config_pair(baseline, changed)
        changed = deepcopy(control)
        changed['stimulus']['contrast_percent'] = 20.0
        with self.assertRaisesRegex(ValueError, 'MAY_ONLY_HALVE_CONTRAST'):
            causality.validate_full_config_pair(baseline, changed)
        changed = deepcopy(control)
        changed['unit_coverage'] = 'model_representatives'
        with self.assertRaisesRegex(ValueError, 'BOTH_FULL_DEVELOPMENT_CONFIGS'):
            causality.validate_full_config_pair(baseline, changed)

    def write_rate_archive(self, root, config, values, *, different_runtime=False, clock_receipt=True):
        (root/'probe').mkdir(parents=True)
        (root/'prepared').mkdir()
        report = dict(status='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT', config=config,
            synthetic_fixture=True, source_units=3, complete_source_table_units=3,
            source_unit_ids=[0, 1, 2], source_models=['toy_a', 'toy_b', 'toy_a'],
            rate_shape=[3, 6], shared_clock_checked_units=3,
            clock_exactly_matches_source_separable_cursor=clock_receipt,
            time_units='s', firing_rate_units='Hz')
        freeze_json(root/'probe/probe.json', report)
        freeze_json(root/'prepared/prepared.json', dict(status='SOURCE_RUNTIME_PREPARED',
                    synthetic_fixture=True, different_runtime=different_runtime))
        with h5py.File(root/'probe/firing_rates_f_tot.h5', 'x') as saved:
            saved.create_dataset('time', data=np.arange(6)/1000.0)
            saved.create_dataset('firing_rates_Hz', data=values)
        relatives = ('probe/probe.json', 'prepared/prepared.json', 'probe/firing_rates_f_tot.h5')
        files = [dict(path=name, bytes=(root/name).stat().st_size, sha256=sha256_file(root/name))
                 for name in relatives]
        freeze_json(root/'delivery.json', dict(status=report['status'], exit_code=0,
                                               synthetic_fixture=True, files=files))

    def fixtures(self, root, *, prefix_differs=False, different_runtime=False, clock_receipt=True):
        baseline = baseline_config()
        baseline['stimulus'].update(duration_s=0.005, gray_screen_s=0.002)
        control = deepcopy(baseline)
        control['stimulus']['contrast_percent'] = 40.0
        control['spike_generation'] = {'starting_noise_seed': 12345, 'trials': 1}
        rates = np.ones((3, 6))
        changed = rates.copy()
        changed[1, 3:] += 0.25
        if prefix_differs:
            changed[2, 2] += 0.125
        first, second, movie_root = root/'baseline', root/'control', root/'movie_archive'
        self.write_rate_archive(first, baseline, rates)
        self.write_rate_archive(second, control, changed, different_runtime=different_runtime,
                                clock_receipt=clock_receipt)
        (movie_root/'movie').mkdir(parents=True)
        movie = dict(status='SOURCE_MOVIE_CONTROL_WITNESS_PASS', synthetic_fixture=True,
            gray_frames=3, first_retained_grating_local_frame=1, source_default_phase=0.0,
            frame_rate_hz=1000.0, movie_shape=[6, 48, 96],
            effective_create_movie_arguments=dict(t_min=0, t_max=0.005, gray_screen_dur=0.002,
                cpd=0.04, temporal_f=2.0, theta=270.0, contrast=0.8),
            checks={name: True for name in ('half_contrast_exact', 'combined_gray_exact',
                'gray_is_zero', 'retained_grating_exact', 'gray_join_frame_count_exact',
                'first_nonzero_frame_follows_gray')})
        freeze_json(movie_root/'movie/report.json', movie)
        path = movie_root/'movie/report.json'
        freeze_json(movie_root/'delivery.json', dict(synthetic_fixture=True,
            files=[dict(path='movie/report.json', bytes=path.stat().st_size, sha256=sha256_file(path))]))
        return first, second, movie_root

    def test_actual_tiny_archives_compare_all_rows_and_preserve_prefix_mismatch(self):
        for prefix_differs in (False, True):
            with self.subTest(prefix_differs=prefix_differs), tempfile.TemporaryDirectory(
                    prefix='synthetic-full-lgn-comparison-') as temporary:
                root = Path(temporary)
                baseline, control, movie = self.fixtures(root, prefix_differs=prefix_differs)
                original_hashes = [sha256_file(path/'probe/firing_rates_f_tot.h5')
                                   for path in (baseline, control)]
                with mock.patch.multiple(causality, SOURCE_UNITS=3, SOURCE_MODELS=2,
                                         RATE_SAMPLES=6, UNCHANGED_MOVIE_FRAMES=3), \
                     mock.patch.object(causality, 'probe_runtime', side_effect=AssertionError('no rate execution')), \
                     mock.patch.object(causality, 'load_prepared_runtime', side_effect=AssertionError('no source loading')):
                    report = causality.compare_full_archives(baseline, control, movie, root/'output')
                self.assertEqual(report['exit_code'], 2 if prefix_differs else 0)
                self.assertEqual(report['status'], 'FULL_SOURCE_CONTROL_PREFIX_DIFFERS_DIAGNOSTIC'
                                 if prefix_differs else 'FULL_SOURCE_CONTROL_PREFIX_EXACT_DEVELOPMENT')
                self.assertEqual(len(report['comparison']['per_unit']), 3)
                self.assertEqual(report['comparison']['unchanged_prefix']['affected_source_ids'],
                                 [2] if prefix_differs else [])
                self.assertEqual(report['comparison']['at_and_after_changed_frame']['changed_source_ids'], [1])
                self.assertFalse(report['comparison']['at_and_after_changed_frame']['response_required_for_every_unit'])
                self.assertEqual(report['rate_config_comparison_excluded_fields'], ['spike_generation'])
                self.assertIn('spike_generation', report['config'])
                self.assertNotIn('spike_generation', report['baseline_config'])
                self.assertFalse(report['rates_recomputed'])
                self.assertEqual(original_hashes, [sha256_file(path/'probe/firing_rates_f_tot.h5')
                                                  for path in (baseline, control)])
                self.assertEqual(report['source_array_references']['control']['sha256'], original_hashes[1])
                persisted = json.loads((root/'output/report.json').read_text())
                self.assertEqual(persisted['exit_code'], report['exit_code'])
                self.assertFalse(persisted['scientific_qualification'])
                self.assertEqual(persisted['production_episodes'], 0)

    def test_archive_runtime_or_every_unit_clock_receipt_difference_is_an_error(self):
        for runtime_differs, clock_receipt in ((True, True), (False, False)):
            with self.subTest(runtime_differs=runtime_differs, clock_receipt=clock_receipt), \
                 tempfile.TemporaryDirectory(prefix='synthetic-full-lgn-receipt-') as temporary:
                root = Path(temporary)
                baseline, control, movie = self.fixtures(root, different_runtime=runtime_differs,
                                                        clock_receipt=clock_receipt)
                with mock.patch.multiple(causality, SOURCE_UNITS=3, SOURCE_MODELS=2,
                                         RATE_SAMPLES=6, UNCHANGED_MOVIE_FRAMES=3):
                    with self.assertRaisesRegex(ValueError, 'PREPARED_RUNTIME_DIFFERS|EVERY_UNIT_CLOCK_RECEIPT'):
                        causality.compare_full_archives(baseline, control, movie, root/'output')
                report = json.loads((root/'output/report.json').read_text())
                self.assertEqual(report['exit_code'], 1)
                self.assertEqual(report['error_type'], 'ValueError')
                self.assertFalse(report['rates_recomputed'])


if __name__ == '__main__':
    unittest.main()
