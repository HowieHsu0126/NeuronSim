import unittest

import numpy as np

from .native_count_windows import NativeEpisodeCounts


class NativeCountWindowTests(unittest.TestCase):
    def test_boundary_counts_preserve_silence_order_and_known_controls(self):
        data = object.__new__(NativeEpisodeCounts)
        data.unit_ids = np.array([3, 7, 12])
        data.event_ticks = np.array([49, 50, 50, 99, 100, 150])
        data.event_units = np.array([0, 0, 1, 1, 0, 1])
        data.pulse_ticks = np.array([50, 100, 150])
        data.bin_ticks = 50
        data.duration_ticks = 200
        count, stimulus = data.window(50, 150)
        np.testing.assert_array_equal(count, [[1, 2, 0], [1, 0, 0]])
        np.testing.assert_array_equal(stimulus, [[1], [1]])
        for a, b in [(49, 150), (50, 201), (100, 50)]:
            with self.assertRaises(ValueError):
                data.window(a, b)


if __name__ == "__main__":
    unittest.main()
