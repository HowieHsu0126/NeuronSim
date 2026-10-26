"""Explicit CUDA observation regressions; not full-scale source qualification."""
import os
import unittest
from unittest import mock

import numpy as np

from . import pfc_deterministic_replay as replay
from .pfc_native_drive import create_drive
from .test_pfc_deterministic_replay import inputs


@unittest.skipUnless(
    os.environ.get("NEURONSIM_PFC_REPLAY_BACKEND") == "cuda",
    "requires explicit CUDA validation invocation",
)
class CudaAppliedCurrentTests(unittest.TestCase):
    def assert_rng_equal(self, before, after):
        self.assertEqual(len(before), len(after))
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_device_current_capture_matches_current_consumed_by_rk2(self):
        config, case, initial, counts_e, counts_i = inputs(4.05)
        config.update(stimulus_start_ms=1.25, stimulus_duration_ms=1.5,
                      stimulus_strength_nA=0.06)
        case["stimulus_strength_nA"] = 0.5
        consumed = []
        original_rk2 = replay._rk2

        def record_current(state, constants, t, dt, arithmetic, population):
            if population == "E":
                self.assertEqual(arithmetic.backend, "cuda")
                self.assertTrue(state["I_stim"].is_cuda)
                consumed.append(arithmetic.host(state["I_stim"]))
            return original_rk2(state, constants, t, dt, arithmetic, population)

        with mock.patch.object(replay, "_rk2", side_effect=record_current):
            result = replay.simulate(config, case, initial, counts_e, counts_i,
                                     backend="cuda", observe_applied_current=True)
        observation = result["applied_current_observation"]
        arrays, metadata = observation["arrays"], observation["metadata"]
        n_e = config["excitatory_neurons"]
        # Native 1-ms operations apply the 1.25--2.75-ms request at 2--3 ms.
        np.testing.assert_array_equal(arrays["segment_start_ticks"], [0, 40, 60])
        np.testing.assert_array_equal(arrays["segment_stop_ticks"], [40, 60, 81])
        np.testing.assert_array_equal(arrays["cue_active"], [False, True, False])
        np.testing.assert_array_equal(arrays["distractor_active"], [False, False, False])
        for key in ("source_unit_indices", "archive_unit_indices"):
            self.assertEqual(arrays[key].dtype, np.dtype(np.int64))
            np.testing.assert_array_equal(arrays[key], np.arange(n_e))
        expected = np.zeros((3, n_e), dtype=np.float64)
        expected[1, [7, 0, 1]] = 0.5 * 1e-9
        self.assertEqual(arrays["current_A"].dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(arrays["current_A"], expected)
        consumed = np.stack(consumed)
        self.assertEqual(consumed.shape, (81, n_e))
        for start, stop, current in zip(arrays["segment_start_ticks"],
                                        arrays["segment_stop_ticks"], arrays["current_A"]):
            np.testing.assert_array_equal(consumed[start:stop],
                                          np.broadcast_to(current, (stop - start, n_e)))
        self.assertEqual(metadata["variable"], "I_stim")
        self.assertEqual(metadata["population"], "E")
        self.assertEqual(metadata["units"], "A")
        self.assertEqual(metadata["timing"], "after_source_stimulus_operation_before_rk2")
        self.assertEqual(metadata["dt_seconds"], config["dt_ms"] / 1000.0)
        self.assertEqual(metadata["stimulus_operation_stride_ticks"], 20)

    def test_observer_preserves_cuda_outputs_and_native_rng(self):
        global_before = np.random.get_state()
        for policy in ("source_global", "independent_initial_external"):
            with self.subTest(policy=policy):
                config, case, _, _, _ = inputs(4.05)
                config.update(stimulus_start_ms=1.0, stimulus_duration_ms=2.0,
                              random_stream_policy=policy)
                case["stimulus_strength_nA"] = 0.5
                if policy == "source_global":
                    config["source_rng_seed"] = 71
                else:
                    config.update(initial_rng_seed=71, external_noise_seed=72)
                initial, plain_drive = create_drive(config, case)
                plain = replay.simulate(config, case, initial, backend="cuda", drive=plain_drive)
                observed_initial, observed_drive = create_drive(config, case)
                observed = replay.simulate(config, case, observed_initial, backend="cuda",
                                           drive=observed_drive, observe_applied_current=True)
                np.testing.assert_array_equal(initial, observed_initial)
                for family in ("raw", "terminal_state"):
                    self.assertEqual(set(plain[family]), set(observed[family]))
                    for key in plain[family]:
                        left, right = plain[family][key], observed[family][key]
                        self.assertEqual(left.dtype, right.dtype)
                        self.assertEqual(left.shape, right.shape)
                        self.assertEqual(left.tobytes(), right.tobytes(), (policy, family, key))
                self.assert_rng_equal(plain_drive.get_state(), observed_drive.get_state())
                self.assert_rng_equal(global_before, np.random.get_state())
                self.assertEqual(plain["inputs_used"], observed["inputs_used"])
                self.assertNotIn("applied_current_observation", plain)
                self.assertEqual(plain["cost"]["applied_current_capture_seconds"], 0.0)
                capture = observed["cost"]["applied_current_capture_seconds"]
                self.assertGreaterEqual(capture, 0.0)
                self.assertLessEqual(capture, observed["cost"]["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
