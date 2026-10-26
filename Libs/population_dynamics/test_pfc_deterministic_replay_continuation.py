"""Terminal continuation parity across source clocks, buffers and suppression."""
from copy import deepcopy
import unittest

import numpy as np

from . import pfc_deterministic_replay as replay
from .pfc_native_drive import create_drive, _state_arrays
from .pfc_coupled_drive import create_coupled_drive
from .test_pfc_deterministic_replay import inputs


class TerminalContinuationTests(unittest.TestCase):
    def assert_arrays_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            a, b = np.asarray(left[key]), np.asarray(right[key])
            self.assertEqual(a.dtype, b.dtype, key)
            self.assertEqual(a.shape, b.shape, key)
            self.assertEqual(a.tobytes(), b.tobytes(), key)

    def assert_outputs_equal(self, left, right):
        for family in ("raw", "terminal_state"):
            self.assert_arrays_equal(left[family], right[family])
        self.assertEqual(left["inputs_used"], right["inputs_used"])

    def settings(self, duration_ms=15.0):
        config, case, initial, e, i = inputs(duration_ms)
        config.update(stimulus_start_ms=0.0, stimulus_duration_ms=10.0, stimulus_width_deg=360.0)
        case["stimulus_strength_nA"] = 1.0
        return config, case, initial, e, i

    def capture_input_blocks(self, drive):
        rows = {p: [] for p in ("E", "I")}
        take = drive.take
        def capture(count):
            block = take(count)
            for p in rows:
                rows[p].append(block[p].copy())
            return block
        drive.take = capture
        return rows

    def resume(self, config, case, prefix, cut, **kwargs):
        return replay.simulate(config, case, None, terminal_state=prefix["terminal_state"],
            start_tick=cut, prefix_raw=prefix["raw"], prefix_inputs_used=prefix["inputs_used"], **kwargs)

    def test_recorded_resume_crosses_input_chunk_and_preserves_all_unit_events(self):
        config, case, initial, e, i = self.settings()
        e[0, 0] = 1000
        i[0, 1] = 1000
        full = replay.simulate(config, case, initial, e, i)
        self.assertGreater(len(full["raw"]["spike_ticks"]), 0)
        for cut in (1, 255, 256, 257):
            with self.subTest(cut=cut):
                prefix = replay.simulate(config, case, initial, e, i, stop_tick=cut)
                before = deepcopy(prefix)
                resumed = self.resume(config, case, prefix, cut, counts_E=e, counts_I=i)
                self.assert_outputs_equal(full, resumed)
                self.assertEqual(resumed["continuation"]["executed_ticks"], 300-cut)
                self.assert_arrays_equal(before["raw"], prefix["raw"])
                self.assert_arrays_equal(before["terminal_state"], prefix["terminal_state"])

    def test_native_resume_restores_both_stream_policies_and_exact_inputs(self):
        for policy in ("source_global", "independent_initial_external"):
            with self.subTest(policy=policy):
                config, case, _, _, _ = self.settings()
                config["random_stream_policy"] = policy
                if policy == "source_global":
                    config["source_rng_seed"] = 71
                else:
                    config.update(initial_rng_seed=71, external_noise_seed=72)
                initial, full_drive = create_drive(config, case)
                full_rows = self.capture_input_blocks(full_drive)
                full = replay.simulate(config, case, initial, drive=full_drive)
                initial, split_drive = create_drive(config, case)
                prefix_rows = self.capture_input_blocks(split_drive)
                prefix = replay.simulate(config, case, initial, drive=split_drive, stop_tick=257)
                saved = replay.snapshot_drive_state(split_drive)
                _, restored = create_drive(config, case)
                suffix_rows = self.capture_input_blocks(restored)
                resumed = self.resume(config, case, prefix, 257, drive=restored, drive_state=saved)
                self.assert_outputs_equal(full, resumed)
                self.assert_arrays_equal(replay.snapshot_drive_state(full_drive), replay.snapshot_drive_state(restored))
                for p in ("E", "I"):
                    self.assert_arrays_equal({p: np.concatenate(full_rows[p])},
                        {p: np.concatenate(prefix_rows[p]+suffix_rows[p])})

    def test_coupled_resume_restores_three_rngs_and_buffered_child_rows(self):
        for factor, cut in ((2, 257), (4, 255), (4, 259)):
            with self.subTest(factor=factor, cut=cut):
                config, case, _, _, _ = self.settings()
                config["source_rng_seed"] = 71
                config["dt_ms"] = 0.05/factor
                config["duration_ms"] = 300*config["dt_ms"]
                initial, full_drive = create_coupled_drive(config, case, 0.05, [81, 82])
                full_rows = self.capture_input_blocks(full_drive)
                full = replay.simulate(config, case, initial, drive=full_drive)
                initial, split_drive = create_coupled_drive(config, case, 0.05, [81, 82])
                prefix_rows = self.capture_input_blocks(split_drive)
                prefix = replay.simulate(config, case, initial, drive=split_drive, stop_tick=cut)
                saved = replay.snapshot_drive_state(split_drive)
                self.assertEqual(int(saved["buffer_offset"]), cut % factor)
                _, restored = create_coupled_drive(config, case, 0.05, [81, 82])
                suffix_rows = self.capture_input_blocks(restored)
                resumed = self.resume(config, case, prefix, cut, drive=restored, drive_state=saved)
                self.assert_outputs_equal(full, resumed)
                self.assert_arrays_equal(replay.snapshot_drive_state(full_drive), replay.snapshot_drive_state(restored))
                for p in ("E", "I"):
                    self.assert_arrays_equal({p: np.concatenate(full_rows[p])},
                        {p: np.concatenate(prefix_rows[p]+suffix_rows[p])})

    def test_legacy_native_rng_snapshot_requires_and_restores_explicit_accounting(self):
        config, case, _, _, _ = self.settings()
        config["source_rng_seed"] = 71
        initial, drive = create_drive(config, case)
        prefix = replay.simulate(config, case, initial, drive=drive, stop_tick=257)
        saved = _state_arrays(drive.get_state())
        _, restored = create_drive(config, case)
        with self.assertRaisesRegex(ValueError, "CURSOR_AND_TOTALS_REQUIRED"):
            replay.restore_drive_state(restored, saved, start_tick=257)
        replay.restore_drive_state(restored, saved, start_tick=257, event_totals=drive.event_totals)
        resumed = self.resume(config, case, prefix, 257, drive=restored)
        initial, full_drive = create_drive(config, case)
        full = replay.simulate(config, case, initial, drive=full_drive)
        self.assert_outputs_equal(full, resumed)
        self.assert_arrays_equal(replay.snapshot_drive_state(full_drive), replay.snapshot_drive_state(restored))

    def test_resume_preserves_focal_suppression_and_current_segments(self):
        config, case, initial, e, i = self.settings()
        initial[0] = -50.0001
        e[0, 0] = 1000
        suppression = replay.FocalSpikeSuppression(0, 0, 270)
        full = replay.simulate(config, case, initial, e, i, focal_spike_suppression=suppression,
                               observe_applied_current=True)
        prefix = replay.simulate(config, case, initial, e, i, stop_tick=257,
            focal_spike_suppression=suppression, observe_applied_current=True)
        self.assertFalse(prefix["suppressed_reference"]["capture_complete"])
        resumed = self.resume(config, case, prefix, 257, counts_E=e, counts_I=i,
            focal_spike_suppression=suppression, prefix_suppressed_reference=prefix["suppressed_reference"],
            observe_applied_current=True, prefix_applied_current_observation=prefix["applied_current_observation"])
        self.assert_outputs_equal(full, resumed)
        self.assert_arrays_equal(full["suppressed_reference"]["forcing"], resumed["suppressed_reference"]["forcing"])
        self.assert_arrays_equal(full["applied_current_observation"]["arrays"], resumed["applied_current_observation"]["arrays"])
        self.assertEqual(full["suppressed_reference"]["conditional_initial"], resumed["suppressed_reference"]["conditional_initial"])
        self.assert_arrays_equal({"ticks": full["suppressed_reference"]["threshold_candidate_ticks"]},
                                 {"ticks": resumed["suppressed_reference"]["threshold_candidate_ticks"]})

    def test_invalid_terminal_identity_event_counts_and_coupled_buffer_fail(self):
        config, case, initial, e, i = self.settings()
        prefix = replay.simulate(config, case, initial, e, i, stop_tick=257)
        for key, replacement in (("E__v", np.zeros(8, np.float32)),
                                 ("E__source_indices", np.arange(8, dtype=np.int32)[::-1])):
            changed = deepcopy(prefix)
            changed["terminal_state"][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "PFC_RESUME_"):
                self.resume(config, case, changed, 257, counts_E=e, counts_I=i)
        changed = deepcopy(prefix)
        changed["raw"]["native_spike_counts"][0] += 1
        with self.assertRaisesRegex(ValueError, "PREFIX_SPIKE_COUNTS_CHANGED"):
            self.resume(config, case, changed, 257, counts_E=e, counts_I=i)
        config["source_rng_seed"] = 71
        config["dt_ms"] = 0.0125
        config["duration_ms"] = 3.75
        initial, drive = create_coupled_drive(config, case, 0.05, [81, 82])
        replay.simulate(config, case, initial, drive=drive, stop_tick=257)
        saved = replay.snapshot_drive_state(drive)
        saved["buffer_offset"] = np.asarray(0)
        _, restored = create_coupled_drive(config, case, 0.05, [81, 82])
        with self.assertRaisesRegex(ValueError, "COUPLED_CURSOR_OR_BUFFER_CHANGED"):
            replay.restore_drive_state(restored, saved)
        self.assertEqual(restored.ticks, 0)

    def test_two_tick_continuation_keeps_native_totals_separate_from_delivered_inputs(self):
        config, case, _, _, _ = self.settings()
        config["source_rng_seed"] = 71
        initial, drive = create_drive(config, case)
        native_take = drive.take
        def override(rows):
            start = drive.ticks
            blocks = native_take(rows)
            if start == 0:
                blocks["E"][0, 0] = 1000
            return blocks
        drive.take = override
        full = replay.simulate(config, case, initial, drive=drive)
        initial, split_drive = create_drive(config, case)
        native_split_take = split_drive.take
        def split_override(rows):
            start = split_drive.ticks
            blocks = native_split_take(rows)
            if start == 0:
                blocks["E"][0, 0] = 1000
            return blocks
        split_drive.take = split_override
        prefix = replay.simulate(config, case, initial, drive=split_drive, stop_tick=298)
        self.assertNotEqual(prefix["inputs_used"]["E"]["events"], split_drive.event_totals["E"])
        _, restored = create_drive(config, case)
        resumed = self.resume(config, case, prefix, 298, stop_tick=300, drive=restored,
                              drive_state=replay.snapshot_drive_state(split_drive))
        self.assert_outputs_equal(full, resumed)
        self.assertEqual(resumed["continuation"]["executed_ticks"], 2)
        self.assertEqual(resumed["continuation"]["reused_prefix_ticks"], 298)
        self.assert_arrays_equal(replay.snapshot_drive_state(drive), replay.snapshot_drive_state(restored))


if __name__ == "__main__":
    unittest.main()
