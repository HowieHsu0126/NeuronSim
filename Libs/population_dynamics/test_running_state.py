"""Causal interval weighting, missingness and source-unit contracts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .running_state import RunningIntervals, align_running_state, read_exported_running_csv


def intervals(start, stop, speed, unit='cm/s'):
    return RunningIntervals(np.asarray(start), np.asarray(stop), np.asarray(speed), unit)


class RunningStateTests(unittest.TestCase):
    def test_unequal_duration_weights_signed_and_absolute_speed_separately(self):
        result = align_running_state(intervals([0.,1.], [1.,4.], [-2.,4.]), [4.], 4.)[0]
        self.assertEqual(result['signed_mean_speed_cm_s'], 2.5)
        self.assertEqual(result['absolute_mean_speed_cm_s'], 3.5)
        self.assertEqual(result['covered_seconds'], 4.)
        self.assertEqual(result['coverage_fraction'], 1.)
        self.assertEqual(result['uncovered_seconds'], 0.)

    def test_gaps_are_reported_and_not_zero_filled(self):
        result = align_running_state(intervals([0.,2.], [1.,3.], [2.,4.]), [4.], 4.)[0]
        self.assertEqual(result['signed_mean_speed_cm_s'], 3.)
        self.assertEqual(result['coverage_fraction'], .5)
        self.assertEqual(result['missing_source_seconds'], 2.)
        self.assertEqual(result['missing_source_intervals'], [
            dict(start_seconds=1.,stop_seconds=2.),dict(start_seconds=3.,stop_seconds=4.)])
        self.assertEqual(result['excluded_intervals'], [])

    def test_future_values_and_future_appended_intervals_cannot_change_alignment(self):
        before = intervals([0.,.5,2.], [.5,1.5,3.], [2.,100.,7.])
        original = align_running_state(before, [1.], 1.)
        changed = intervals([0.,.5,2.,4.], [.5,1.5,3.,5.], [2.,-1e12,1e15,-8e10])
        self.assertEqual(original, align_running_state(changed,[1.],1.))
        result = original[0]
        self.assertEqual(result['signed_mean_speed_cm_s'], 2.)
        self.assertEqual(result['coverage_fraction'], .5)
        self.assertEqual(result['excluded_intervals'][0]['reasons'], ['ends_after_onset'])
        self.assertEqual(result['excluded_interval_seconds'], .5)

    def test_history_boundary_crossing_interval_is_not_clipped_into_mean(self):
        result = align_running_state(intervals([-.5,.25,.75], [.25,.75,1.], [100.,2.,4.]), [1.], 1.)[0]
        self.assertAlmostEqual(result['signed_mean_speed_cm_s'], 8./3.)
        self.assertEqual(result['covered_seconds'], .75)
        self.assertEqual(result['excluded_interval_seconds'], .25)
        self.assertEqual(result['excluded_intervals'][0]['reasons'], ['crosses_history_start'])

    def test_exact_cutoff_and_history_boundaries_use_half_open_support(self):
        result = align_running_state(intervals([-1.,0.,.5,1.], [0.,.5,1.,2.], [999.,2.,4.,999.]), [1.], 1.)[0]
        self.assertEqual(result['included_interval_indices'], [1,2])
        self.assertEqual(result['signed_mean_speed_cm_s'], 3.)
        self.assertEqual(result['coverage_fraction'], 1.)
        self.assertEqual(result['excluded_intervals'], [])

    def test_interval_crossing_both_boundaries_counts_missing_duration_once(self):
        result = align_running_state(intervals([0.], [10.], [3.]), [5.], 2.)[0]
        self.assertIsNone(result['signed_mean_speed_cm_s'])
        self.assertIsNone(result['absolute_mean_speed_cm_s'])
        self.assertEqual(result['coverage_fraction'], 0.)
        self.assertEqual(result['missing_source_seconds'], 0.)
        self.assertEqual(result['excluded_interval_seconds'], 2.)
        self.assertEqual(result['uncovered_seconds'], 2.)
        self.assertEqual(result['excluded_intervals'][0]['reasons'], ['crosses_history_start','ends_after_onset'])

    def test_explicit_invalid_quality_interval_retains_exclusion(self):
        result = align_running_state(intervals([0.,1.], [1.,2.], [2.,4.]), [2.], 2.,
                                     valid_intervals=np.array([True,False]))[0]
        self.assertEqual(result['signed_mean_speed_cm_s'], 2.)
        self.assertEqual(result['coverage_fraction'], .5)
        self.assertEqual(result['excluded_intervals'][0]['reasons'], ['source_marked_invalid'])
        self.assertEqual(result['excluded_interval_seconds'], 1.)

    def test_empty_source_or_unobserved_window_has_missing_value_not_zero_speed(self):
        for source in [intervals([],[],[]), intervals([5.],[6.],[2.])]:
            result = align_running_state(source,[2.],1.)[0]
            self.assertIsNone(result['signed_mean_speed_cm_s'])
            self.assertIsNone(result['absolute_mean_speed_cm_s'])
            self.assertEqual(result['missing_source_seconds'],1.)
            self.assertEqual(result['coverage_fraction'],0.)

    def test_duration_accounting_and_caller_onset_order(self):
        source = intervals([.1,.7,1.4,2.1],[.3,1.1,1.5,2.3],[-2.,3.,-1.,4.])
        onsets=[3.,.5,2.25,1.]
        results=align_running_state(source,onsets,1.)
        self.assertEqual([row['onset_seconds'] for row in results],onsets)
        for row in results:
            self.assertAlmostEqual(row['covered_seconds']+row['uncovered_seconds'],row['window_duration_seconds'],places=14)
            self.assertGreaterEqual(row['coverage_fraction'],0.)
            self.assertLessEqual(row['coverage_fraction'],1.)

    def test_invalid_interval_records_are_rejected(self):
        sources = [
            intervals([0.],[0.],[1.]), intervals([1.],[0.],[1.]),
            intervals([0.,.5],[1.,1.5],[1.,2.]),
            intervals([1.,0.],[2.,1.],[1.,2.]),
            intervals([np.nan],[1.],[1.]), intervals([0.],[np.inf],[1.]),
            intervals([0.],[1.],[np.nan]), intervals([0.],[1.],[np.inf]),
            intervals([0.],[1.],[]), intervals([False],[True],[1.]),
            intervals([0.],[1.],[1.],unit='m/s'),
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                align_running_state(source,[1.],1.)

    def test_invalid_history_onset_or_quality_mask_is_rejected(self):
        source=intervals([0.],[1.],[2.])
        for history in [0.,-1.,np.inf,np.nan,True]:
            with self.subTest(history=history),self.assertRaises(ValueError):
                align_running_state(source,[1.],history)
        for onset in [np.inf,np.nan]:
            with self.assertRaises(ValueError):align_running_state(source,[onset],1.)
        with self.assertRaises(ValueError):
            align_running_state(source,[1.],1.,valid_intervals=[1])
        with self.assertRaises(ValueError):
            align_running_state(source,[1.],1.,valid_intervals=[True,False])

    def test_exported_physical_values_are_not_converted_twice(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'running.csv'
            path.write_text('start_time,value,value_is_finite,stop_time\n0,3,True,1\n',encoding='utf-8')
            source=read_exported_running_csv(path,dict(unit='cm/s',conversion=2.,offset=10.))
            self.assertEqual(align_running_state(source,[1.],1.)[0]['signed_mean_speed_cm_s'],3.)
            with self.assertRaisesRegex(ValueError,'PHYSICAL_UNIT'):
                read_exported_running_csv(path,dict(unit='m/s'))
            path.write_text('start_time,value,value_is_finite,stop_time\n0,nan,False,1\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'NONFINITE'):
                read_exported_running_csv(path,dict(unit='cm/s'))


if __name__ == '__main__':
    unittest.main()
