"""Independent control enumeration rejects early or mis-targeted TC output."""
import copy
import unittest

import numpy as np

from .trajectory_data import verify_controls


class TrajectoryControlTests(unittest.TestCase):
    def test_source_mask_clock_multiplicity_and_explicit_omission(self):
        source = {"resolved_source_network": {"th_radius": .5, "th_delay_pulse_generator": 1.}}
        case = {"settings": {"pulse_control": {"initial_emission_ms": [99., 149.]}}}
        events = dict(unit_ids=np.array([1, 2, 3]), populations=np.array(["L23E", "TC", "TC"]),
                      positions_mm=np.array([[0., 0.], [.2, 0.], [.8, 0.]]),
                      spike_unit_ids=np.array([1, 2, 2]), spike_ticks=np.array([999, 1000, 1500]))
        indices = np.array([0, 1, 1])
        receipt = verify_controls(events, indices, case, source, .1)
        self.assertEqual(receipt["tc_arrival_ticks"], [1000, 1500])
        self.assertEqual(receipt["active_tc_units"], 1)
        wrong = copy.deepcopy(events); wrong["spike_ticks"][1] -= 1
        with self.assertRaisesRegex(ValueError, "TC events differ"):
            verify_controls(wrong, indices, case, source, .1)
        wrong = copy.deepcopy(events); wrong["spike_unit_ids"][1] = 3
        with self.assertRaisesRegex(ValueError, "TC events differ"):
            verify_controls(wrong, np.array([0, 2, 1]), case, source, .1)
        omitted = copy.deepcopy(events)
        for key in ("spike_unit_ids", "spike_ticks"):
            omitted[key] = omitted[key][:1]
        omission = {"settings": {"pulse_control": {"initial_emission_ms": []}}}
        self.assertEqual(verify_controls(omitted, np.array([0]), omission, source, .1)["tc_events"], 0)


if __name__ == "__main__":
    unittest.main()
