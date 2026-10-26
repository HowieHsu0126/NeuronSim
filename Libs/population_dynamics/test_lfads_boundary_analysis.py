"""Crossing algebra and complete-coverage evidence checks on tiny CSV fixtures."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import tempfile
import unittest

import yaml

from .lfads_boundary_analysis import BRANCHES, crossed_contrasts, metric_names, read_rows, source_binding
from .provenance import sha256_file


class BoundaryAnalysisTests(unittest.TestCase):
    def test_complete_evaluated_identity_preserves_and_checks_training_subset(self):
        trained = dict(acquisition_design_sha256='a', dataset_sha256='b',
            native_source_identity=dict(native_library_sha256='c', graph_manifest_sha256='d', native_poisson_rng_family='philox'),
            config=dict(dt_ms=.1))
        native = dict(trained['native_source_identity'], gpu_source_revision='source', dt_ms=.1)
        report = dict(training_binding=trained, acquisition_design_sha256='a', dataset_sha256='b', native_source_identity=native)
        self.assertEqual(source_binding(report)['native_source_identity'], native)
        for location, key, value in [('native_source_identity', 'native_library_sha256', 'other'),
                                     ('native_source_identity', 'dt_ms', .2)]:
            changed = copy.deepcopy(report); changed[location][key] = value
            with self.assertRaises(ValueError): source_binding(changed)
        changed = copy.deepcopy(report); changed['dataset_sha256'] = 'other'
        with self.assertRaises(ValueError): source_binding(changed)

    def test_paired_factorial_changes_and_interaction(self):
        values = dict(zip(BRANCHES, [10., 7., 6., 1.]))
        self.assertEqual(crossed_contrasts(values), dict(boundary_at_prior=-3., boundary_at_posterior=-5.,
                         future_at_prefix=-4., future_at_full_response=-6., boundary_future_interaction=-2.))
        self.assertEqual(crossed_contrasts(dict.fromkeys(BRANCHES)), {})
        for bad in ({BRANCHES[0]: 1.}, dict(zip(BRANCHES, [1., 2., None, 4.]))):
            with self.assertRaises(ValueError):
                crossed_contrasts(bad)

    def fixture(self, folder):
        config = yaml.safe_load((Path(__file__).parents[1]/'configs/neural_population_lfads_boundary_diagnostic.yaml').read_text())
        report = dict(configuration=config, status='DIAGNOSED_DEVELOPMENT', model_type='lfads_boundary_diagnostic',
                      completed_cases=16, completed_branches=64, score_rows=192, training_seed=41,
                      formal_test_accessed=False, calibration_outcomes_accessed=False, weights_updated=False,
                      scientific_claim_qualified=False, artifacts={}, endpoint_checks=[])
        rows = []
        for block in range(14, 18):
            for template in range(4):
                case = f'train_block_{block:03d}_familiar_{template}'
                for branch in BRANCHES:
                    name = f'{case}_{branch}.npz'
                    report['artifacts'][name] = 'fixture_not_loaded_by_row_parser'
                    for ms in config['horizons_ms']:
                        row = dict(seed=41, block=block, template_id=f'familiar_{template}', case_id=case,
                                   branch=branch, role='forecast' if branch == BRANCHES[0] else 'response_conditioned_diagnostic',
                                   origin_ms=1000., horizon_ms=ms, ensemble_artifact=name)
                        row.update({metric: 1. for metric in metric_names(config)})
                        for lag in config['temporal_lags_bins']:
                            if lag >= int(ms/config['bin_ms']):
                                row[f'variogram_temporal_lag_{lag}'] = None
                        rows.append(row)
                    if branch in (BRANCHES[0], BRANCHES[3]):
                        report['endpoint_checks'].append(dict(case_id=case, branch=branch, all_within_tolerance=True,
                            rtol=config['endpoint_rtol'], atol=config['endpoint_atol'], sampled_counts_bitwise_equal=True,
                            arrays={k: dict(within_tolerance=True, bitwise_equal=True) for k in
                                    ('conditional_expected_counts', 'boundary_generator_states', 'future_generator_states')}))
        self.write_rows(folder, rows, report)
        return rows, report

    @staticmethod
    def write_rows(folder, rows, report):
        with (folder/'scores.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        report['scores_sha256'] = sha256_file(folder/'scores.csv')

    def test_complete_rows_preserve_optional_lags(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); _, report = self.fixture(folder)
            rows = read_rows(folder, report)
            self.assertEqual(len(rows), 192)
            first = rows[('train_block_014_familiar_0', BRANCHES[0], 20.)]
            self.assertIsNone(first['variogram_temporal_lag_4'])
            self.assertEqual(first['absolute_conditional_mean_error'], 0.)

    def test_missing_duplicate_and_role_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); rows, report = self.fixture(folder)
            for changed in (rows[:-1], rows[:-1]+[rows[0]]):
                self.write_rows(folder, changed, report)
                with self.assertRaises(ValueError):
                    read_rows(folder, report)
            changed = copy.deepcopy(rows); changed[3]['role'] = 'forecast'
            self.write_rows(folder, changed, report)
            with self.assertRaisesRegex(ValueError, 'role'):
                read_rows(folder, report)

    def test_failed_or_relaxed_endpoint_cannot_enter_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); _, report = self.fixture(folder)
            for key, value in [('rtol', 0.01), ('sampled_counts_bitwise_equal', False)]:
                changed = copy.deepcopy(report); changed['endpoint_checks'][0][key] = value
                with self.assertRaises(ValueError):
                    read_rows(folder, changed)
            changed = copy.deepcopy(report)
            changed['configuration']['endpoint_rtol'] = 0.01
            for check in changed['endpoint_checks']:
                check['rtol'] = 0.01
            with self.assertRaises(ValueError):
                read_rows(folder, changed)


if __name__ == '__main__':
    unittest.main()
