import json
from pathlib import Path
import tempfile
import unittest

from .carlsim_event_order import compare_events, diagnose
from .carlsim_prefix_replay import compare
from .carlsim_records import _sha, validate_prefix
from .test_carlsim_prefix_replay import fixture
from .test_carlsim_records import events


class EventOrder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a, self.b = self.root/'a.dat', self.root/'b.dat'

    def tearDown(self):
        self.temp.cleanup()

    def check(self, a, b):
        events(self.a, 3, a)
        events(self.b, 3, b)
        return compare_events(self.a, self.b, units=3, duration_ms=30)

    def test_same_time_permutations_retain_duplicate_multiplicity(self):
        r = self.check([(1, 0), (1, 1), (1, 0), (2, 2)],
                       [(1, 1), (1, 0), (1, 0), (2, 2)])
        self.assertTrue(r['event_multiset_equal'])
        self.assertTrue(r['timestamp_sequence_equal'])
        self.assertFalse(r['event_order_equal'])
        self.assertEqual(r['differing_file_rows'], 2)

    def test_identical_marginals_do_not_hide_time_identity_change(self):
        r = self.check([(1, 0), (2, 1)], [(1, 1), (2, 0)])
        self.assertFalse(r['event_multiset_equal'])

    def test_same_unique_pairs_do_not_hide_duplicate_count_change(self):
        r = self.check([(1, 0), (1, 0), (1, 1)], [(1, 0), (1, 1), (1, 1)])
        self.assertFalse(r['event_multiset_equal'])
        r = self.check([(1, 0), (1, 0)], [(1, 0)])
        self.assertFalse(r['event_multiset_equal'])
        self.assertIsNone(r['differing_file_rows'])

    def test_time_order_is_reported_separately_and_empty_events_work(self):
        r = self.check([(2, 0), (1, 1)], [(1, 1), (2, 0)])
        self.assertTrue(r['event_multiset_equal'])
        self.assertFalse(r['timestamp_sequence_equal'])
        self.assertTrue(self.check([], [])['event_order_equal'])

    def test_bounds_and_truncation_are_rejected(self):
        events(self.a, 3, [(30, 0)])
        events(self.b, 3, [])
        with self.assertRaisesRegex(ValueError, 'OUT_OF_RANGE'):
            compare_events(self.a, self.b, units=3, duration_ms=30)
        self.a.write_bytes(self.b.read_bytes() + b'abc')
        with self.assertRaisesRegex(ValueError, 'TRUNCATED_EVENT'):
            compare_events(self.a, self.b, units=3, duration_ms=30)

    def test_diagnosis_is_bound_to_strict_failure_and_does_not_overwrite_it(self):
        left, right = self.root/'left', self.root/'right'
        fixture(left)
        fixture(right, original=left)
        events(right/'native_run/results/spk_E.dat', 3, [(20, 2), (2, 0), (20, 2), (4449, 1)])
        # Preserve original decoder output and regenerate a distinct current observation.
        (right/'observations').rename(right/'before_event_order')
        validate_prefix(right, right/'observations')
        strict = self.root/'strict.json'
        with self.assertRaisesRegex(ValueError, 'repeat differs'):
            compare(left, right, strict)
        before = strict.read_bytes()
        out = self.root/'order.json'
        result = diagnose(left, right, strict, out)
        self.assertEqual(result['status'], 'EVENT_TIME_ID_MULTISETS_EQUAL_TIMESTAMP_ORDER_DIFFERS')
        self.assertEqual(strict.read_bytes(), before)
        self.assertFalse(result['scientific_fidelity_qualified'])
        with self.assertRaises(FileExistsError):
            diagnose(left, right, strict, out)
        events(right/'native_run/results/spk_E.dat', 3, [(20, 1)])
        with self.assertRaisesRegex(ValueError, 'spike file differs'):
            diagnose(left, right, strict, self.root/'changed.json')


if __name__ == '__main__':
    unittest.main()
