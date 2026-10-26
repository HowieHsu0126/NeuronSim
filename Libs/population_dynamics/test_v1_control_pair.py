"""Tiny synthetic spike archives; no sampling or neural simulation is performed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import v1_control_pair as pair_module
from .provenance import freeze_json, sha256_file
from .v1_lgn_spikes import load_spike_trial, write_spike_trial


def config(contrast):
    return dict(scope='development_source_runtime_qualification', unit_coverage='all_source_units',
        converter_python='/usr/bin/python3',
        stimulus=dict(duration_s=3.0, gray_screen_s=0.5, spatial_frequency_cycles_per_degree=0.04,
                      temporal_frequency_hz=2.0, direction_deg=270.0, contrast_percent=contrast),
        spike_generation=dict(starting_noise_seed=12345, trials=1))


def delivery(root, relatives):
    freeze_json(root/'delivery.json', dict(synthetic_fixture=True,
        files=[dict(path=name, bytes=(root/name).stat().st_size, sha256=sha256_file(root/name))
               for name in relatives]))


def fixture(root, *, changed_time=False, count_difference=False, seed_difference=False,
            branch_difference=False, bad_rate_identity=False, bad_config=False):
    ids = np.arange(3, dtype=np.int64)
    left = [np.array([1., 1., 500., 501., 800., 3000.]), np.array([], dtype=np.float64), np.array([2., 600.])]
    right = [np.array([1., 1., 500., 501., 900., 3000.]), np.array([], dtype=np.float64), np.array([2., 600.])]
    if changed_time:
        right[0][2] = 499.
    if count_difference:
        right[0] = right[0][1:]
    seeds = [dict(source_id=int(i), primary_seed=100+int(i), poisson_seed=None, sampler='bernoulli') for i in ids]
    other_seeds = deepcopy(seeds)
    if seed_difference:
        other_seeds[0]['primary_seed'] = 999
    if branch_difference:
        other_seeds[2].update(poisson_seed=777, sampler='poisson')
    paths = {}
    for role, trains, recorded_seeds, contrast, rate_report_hash, rate_file_hash in (
            ('baseline', left, seeds, 80.0, 'a'*64, 'b'*64),
            ('control', right, other_seeds, 40.0, 'c'*64, 'd'*64)):
        archive = root/role
        (archive/'spikes').mkdir(parents=True)
        trial = write_spike_trial(ids, trains, recorded_seeds, 3000.0, archive/'spikes/trial_0000')
        settings = config(contrast)
        if role == 'control' and bad_config:
            settings['stimulus']['direction_deg'] = 90.0
        report = dict(status='SOURCE_SPIKES_EXPORTED_EXACT', synthetic_fixture=True,
            config=settings, source_units=3, rate_shape=[3, 3001], starting_noise_seed=12345,
            rate_report_sha256=rate_report_hash,
            rate_file_sha256='9'*64 if role == 'control' and bad_rate_identity else rate_file_hash,
            prepared_runtime_sha256='e'*64, trials=[dict(trial=0, **trial)],
            artifacts={name: sha256_file(archive/'spikes'/name) for name in (trial['plan'], trial['sonata'])})
        freeze_json(archive/'spikes/report.json', report)
        delivery(archive, ['spikes/report.json', 'spikes/'+trial['plan'], 'spikes/'+trial['sonata']])
        paths[role] = archive
    rate_root = root/'rate_pair'
    (rate_root/'full_causality').mkdir(parents=True)
    rate_pair = dict(status='FULL_SOURCE_CONTROL_PREFIX_EXACT_DEVELOPMENT', exit_code=0,
        synthetic_fixture=True, source_units=3, source_unit_ids=[0, 1, 2], changed_input_frame=501,
        every_unit_clock_receipts_verified=True, baseline_config=config(80.0), config=config(40.0),
        baseline_report_sha256='a'*64, control_report_sha256='c'*64,
        prepared_runtime_hashes={'baseline': 'e'*64, 'control': 'e'*64}, prepared_runtime_hashes_exact=True,
        source_array_references={'baseline': {'shape': [3, 3001], 'sha256': 'b'*64},
                                 'control': {'shape': [3, 3001], 'sha256': 'd'*64}},
        comparison=dict(source_units=3, clock_exact=True,
            unchanged_prefix=dict(exact=True, mismatch_count=0, max_absolute_difference_hz=0.0,
                affected_source_ids=[], frame_start=0, frame_stop_exclusive=501),
            at_and_after_changed_frame=dict(frame_start=501, first_time_s=0.501),
            per_unit=[dict(source_id=int(i), prefix_exact=True, prefix_mismatch_count=0,
                           prefix_max_absolute_difference_hz=0.0) for i in ids]))
    freeze_json(rate_root/'full_causality/report.json', rate_pair)
    delivery(rate_root, ['full_causality/report.json'])
    return paths['baseline'], paths['control'], rate_root


class MatchedInputTests(unittest.TestCase):
    def run_fixture(self, root, **options):
        baseline, control, rate_pair = fixture(root, **options)
        before = [sha256_file(path/'spikes/trial_0000.npz') for path in (baseline, control)]
        with mock.patch.object(pair_module, 'SOURCE_UNITS', 3):
            report = pair_module.compare_archives(baseline, control, rate_pair, root/'output')
        self.assertEqual(before, [sha256_file(path/'spikes/trial_0000.npz') for path in (baseline, control)])
        persisted = json.loads((root/'output/report.json').read_text())
        self.assertEqual(persisted['exit_code'], report['exit_code'])
        return report

    def test_exact_prefix_preserves_duplicates_silent_units_and_later_change(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-spike-pair-') as temporary:
            report = self.run_fixture(Path(temporary))
        self.assertEqual(report['status'], 'SOURCE_SPIKE_CONTROL_PAIR_EXACT_DEVELOPMENT')
        self.assertEqual(report['exit_code'], 0)
        comparison = report['comparison']
        self.assertTrue(comparison['recorded_sampler_initialization']['exact'])
        self.assertEqual(comparison['prefix']['global_comparison']['expected_events'], 4)
        self.assertEqual(comparison['prefix']['global_comparison']['mapped_generators'], 3)
        self.assertEqual(comparison['later']['changed_source_ids'], [0])
        self.assertEqual(comparison['per_unit'][0]['prefix']['baseline_count'], 3)
        self.assertEqual(comparison['per_unit'][0]['later']['baseline_count'], 3)
        self.assertEqual(comparison['per_unit'][1]['prefix']['baseline_count'], 0)
        self.assertEqual(comparison['per_unit'][1]['later']['control_count'], 0)
        self.assertFalse(report['hidden_random_draws_equal_claimed'])
        self.assertFalse(report['per_unit_noise_independence_claimed'])

    def test_one_changed_prefix_timestamp_is_a_valid_differing_pair(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-spike-time-') as temporary:
            report = self.run_fixture(Path(temporary), changed_time=True)
        self.assertEqual(report['exit_code'], 2)
        self.assertEqual(report['comparison']['prefix']['changed_source_ids'], [0])
        detail = report['comparison']['per_unit'][0]['prefix']
        self.assertTrue(detail['counts_exact'])
        self.assertEqual(detail['timestamp_mismatches'], 1)
        self.assertEqual(detail['max_absolute_time_difference_ms'], 1.0)

    def test_different_counts_leave_time_pair_metrics_undefined(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-spike-count-') as temporary:
            report = self.run_fixture(Path(temporary), count_difference=True)
        self.assertEqual(report['exit_code'], 2)
        detail = report['comparison']['per_unit'][0]['prefix']
        self.assertEqual((detail['baseline_count'], detail['control_count']), (3, 2))
        self.assertIsNone(detail['timestamp_mismatches'])
        self.assertIsNone(detail['max_absolute_time_difference_ms'])
        self.assertIsNone(report['comparison']['prefix']['global_comparison']['timestamp_mismatches'])

    def test_same_starting_seed_does_not_hide_recorded_seed_or_branch_differences(self):
        for option, identity, fields in (('seed_difference', 0, ['source_primary_seed']),
                ('branch_difference', 2, ['source_poisson_seed', 'source_sampler_code'])):
            with self.subTest(option=option), tempfile.TemporaryDirectory(prefix='synthetic-spike-seed-') as temporary:
                report = self.run_fixture(Path(temporary), **{option: True})
            self.assertEqual(report['exit_code'], 2)
            self.assertTrue(report['comparison']['prefix']['exact'])
            self.assertEqual(report['comparison']['recorded_sampler_initialization']['changed_source_ids'], [identity])
            self.assertEqual(report['comparison']['per_unit'][identity]['initialization_changed_fields'], fields)
            self.assertEqual(report['baseline_config']['spike_generation']['starting_noise_seed'], 12345)
            self.assertEqual(report['control_config']['spike_generation']['starting_noise_seed'], 12345)

    def test_mismatched_rate_or_configuration_identity_rejects(self):
        for option in ('bad_rate_identity', 'bad_config'):
            with self.subTest(option=option), tempfile.TemporaryDirectory(prefix='synthetic-spike-binding-') as temporary:
                root = Path(temporary)
                baseline, control, rate_pair = fixture(root, **{option: True})
                with mock.patch.object(pair_module, 'SOURCE_UNITS', 3):
                    with self.assertRaisesRegex(ValueError, 'RATE_SOURCE_IDENTITY_DIFFERS|CONFIG_MAY_ONLY_CHANGE'):
                        pair_module.compare_archives(baseline, control, rate_pair, root/'output')
                report = json.loads((root/'output/report.json').read_text())
                self.assertEqual(report['exit_code'], 1)
                self.assertFalse(report['new_sampling'])

    def test_complete_domain_dtype_sentinel_and_duplicate_ids_validate_before_prefix(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-spike-schema-') as temporary:
            baseline, _, _ = fixture(Path(temporary))
            plan, _ = load_spike_trial(baseline)
            cases = []
            changed = deepcopy(plan)
            changed['source_event_time_ms'][-1] = 3000.25
            cases.append(changed)
            changed = deepcopy(plan)
            changed['source_primary_seed'] = changed['source_primary_seed'].astype(np.float64)
            cases.append(changed)
            changed = deepcopy(plan)
            changed['source_poisson_seed'][0] = 10
            cases.append(changed)
            changed = deepcopy(plan)
            changed['source_sampler_code'][0] = 1
            cases.append(changed)
            changed = deepcopy(plan)
            changed['source_id'][1] = 0
            cases.append(changed)
            with mock.patch.object(pair_module, 'SOURCE_UNITS', 3):
                for index, changed in enumerate(cases):
                    with self.subTest(index=index), self.assertRaises(ValueError):
                        pair_module.validate_source_plan(changed, np.arange(3, dtype=np.int64))


if __name__ == '__main__':
    unittest.main()
