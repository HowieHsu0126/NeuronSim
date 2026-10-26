"""Check source schedule identity before any live native library mutation."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from . import nest_gpu_live_control as control
from .mesocircuit_continuation import prepare_pulse_protocol

NODES = (21, 22, 23)
MASK = (True, True, False)
INITIAL_EMISSIONS = [2.0, 9.0, 9.5, 10.0, 10.1, 11.0, 14.0]
# Native host spike_times encode folded TC events one dt before their recorded
# arrival. These are fixture values for arrivals 3, 10, 10.5, 11, 11.1, 12, 15 ms.
INITIAL_HOST_TIMES = np.asarray([2.9, 9.9, 10.4, 10.9, 11.0, 11.9, 14.9], dtype=np.float32)


def protocol(initial=None, *, future=None):
    settings = dict(initial_emission_ms=INITIAL_EMISSIONS if initial is None else initial,
                    decision_time_ms=10.0)
    if future is not None:
        settings["future_emission_ms"] = future
    return prepare_pulse_protocol(
        dict(thalamic_input=True, thalamic_input_type="pulses", th_pulse_start=3.0,
             th_interval=7.0, th_delay_pulse_generator=1.0),
        dict(sim_resolution=0.1, t_presim=0.0, t_sim=20.0), settings)


class NativeNodeSeq:
    """Integer indexing and length match the pinned native NodeSeq wrapper."""

    def __init__(self, first, count):
        self.i0, self.n = first, count
        self.indices = []

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self.indices.append(index)
        if index < -self.n:
            raise ValueError("Sequence index error")
        if index >= self.n:
            raise ValueError("Sequence index out of range")
        if index < 0:
            index = index % self.n
        return self.i0 + index


class NativeScheduleReadback:
    """A read-only backend: native library loading must remain unnecessary."""

    def __init__(self, schedules=None):
        self.schedules = ({21: INITIAL_HOST_TIMES.copy(), 22: INITIAL_HOST_TIMES.copy(),
                           23: np.empty(0, dtype=np.float32)} if schedules is None else schedules)
        self.reads = []

    def GetNeuronStatus(self, nodes, field):
        if field != "spike_times" or len(nodes) != 1:
            raise AssertionError("one generator spike_times identity read required")
        node = int(nodes[0])
        self.reads.append(node)
        return [np.asarray(self.schedules[node], dtype=np.float32).tolist()]

    def GetErrorCode(self):
        return 0

    def GetErrorMessage(self):
        return ""


class NativeControlIdentityTests(unittest.TestCase):
    def test_unchanged_schedule_verifies_all_sources_without_loading_native_library(self):
        backend = NativeScheduleReadback()
        with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
            receipt = control.replace_masked_tc_future(backend, NODES, MASK, protocol())
        loader.assert_not_called()
        self.assertEqual(backend.reads, list(NODES))
        self.assertTrue(receipt["initial_tc_schedule_verified"])
        self.assertFalse(receipt["future_schedule_replaced"])
        self.assertFalse(receipt["scientific_intervention_qualified"])

    def test_native_NodeSeq_unchanged_schedule_uses_only_declared_indices(self):
        backend = NativeScheduleReadback()
        nodes = NativeNodeSeq(NODES[0], len(NODES))
        with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
            receipt = control.replace_masked_tc_future(backend, nodes, MASK, protocol())
        loader.assert_not_called()
        self.assertEqual(nodes.indices, [0, 1, 2])
        self.assertEqual(backend.reads, list(NODES))
        self.assertTrue(receipt["initial_tc_schedule_verified"])
        self.assertFalse(receipt["future_schedule_replaced"])

    def test_native_NodeSeq_replacement_verifies_all_sources_before_active_writes(self):
        backend = NativeScheduleReadback()
        nodes = NativeNodeSeq(NODES[0], len(NODES))
        replacement = protocol(future=[12.0])
        mutations = []

        def replace(node, count, ticks_pointer, decision_tick):
            ticks = np.asarray([ticks_pointer[i] for i in range(count)], dtype=np.int64)
            mutations.append((node, ticks.copy(), decision_tick, backend.reads.copy()))
            backend.schedules[node] = ((ticks - 1) * np.float32(replacement.resolution_ms)).astype(np.float32)
            return 0

        library = mock.Mock()
        library.NeuronSim_ReplaceFutureSpikeGenerator.side_effect = replace
        with mock.patch.object(control.ctypes, "CDLL", return_value=library) as loader, \
                mock.patch.dict(control.os.environ, {"NESTGPU_LIB": "fixture.so"}):
            receipt = control.replace_masked_tc_future(backend, nodes, MASK, replacement)
        loader.assert_called_once_with("fixture.so")
        self.assertEqual(nodes.indices, [0, 1, 2, 0, 1, 2])
        self.assertEqual([item[0] for item in mutations], [21, 22])
        self.assertEqual(mutations[0][3], list(NODES))
        for _, ticks, decision_tick, _ in mutations:
            np.testing.assert_array_equal(ticks, [105, 110, 130])
            self.assertEqual(decision_tick, 100)
        self.assertEqual(backend.reads, [21, 22, 23, 21, 22])
        self.assertEqual(backend.schedules[23].size, 0)
        self.assertEqual(receipt["updated_generators"], 2)
        self.assertTrue(receipt["initial_tc_schedule_verified"])
        self.assertTrue(receipt["device_schedule_readback_verified"])
        self.assertFalse(receipt["scientific_intervention_qualified"])

    def test_empty_initial_schedule_keeps_active_mask_identity_without_loading_library(self):
        backend = NativeScheduleReadback({node: np.empty(0, dtype=np.float32) for node in NODES})
        with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
            receipt = control.replace_masked_tc_future(backend, NODES, MASK, protocol(initial=[]))
        loader.assert_not_called()
        self.assertEqual(backend.reads, list(NODES))
        self.assertTrue(receipt["initial_tc_schedule_verified"])
        self.assertEqual(receipt["remaining_tc_arrival_ticks"], [])

    def test_changed_consumed_inflight_or_future_chronology_rejects_before_library_load(self):
        # Checking only remaining arrivals would miss removal of consumed events.
        # The source identity must bind the entire original control program.
        altered = {
            "consumed_event_removed": np.delete(INITIAL_HOST_TIMES, 0),
            "inflight_event_removed": np.delete(INITIAL_HOST_TIMES, 2),
            "future_event_changed": np.asarray([2.9, 9.9, 10.4, 10.9, 11.0, 12.9, 14.9], dtype=np.float32),
            "chronology_reversed": INITIAL_HOST_TIMES[::-1],
        }
        for name, host_times in altered.items():
            for future in (None, []):
                with self.subTest(case=name, mutation_requested=future is not None):
                    backend = NativeScheduleReadback()
                    backend.schedules[21] = host_times
                    with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
                        with self.assertRaises(ValueError):
                            control.replace_masked_tc_future(backend, NODES, MASK, protocol(future=future))
                    loader.assert_not_called()

    def test_wrong_active_mask_rejects_before_library_load(self):
        # Node 21 really has the declared pulse schedule; node 23 is silent.
        # Swapping these mask entries must fail even without a future mutation.
        for future in (None, []):
            with self.subTest(mutation_requested=future is not None):
                backend = NativeScheduleReadback()
                with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
                    with self.assertRaises(ValueError):
                        control.replace_masked_tc_future(backend, NODES, (False, True, True), protocol(future=future))
                loader.assert_not_called()

    def test_late_source_mismatch_prevents_any_native_modification(self):
        backend = NativeScheduleReadback()
        backend.schedules[23] = np.asarray([2.9], dtype=np.float32)
        with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
            with self.assertRaises(ValueError):
                control.replace_masked_tc_future(backend, NODES, MASK, protocol(future=[12.0]))
        loader.assert_not_called()
        self.assertEqual(backend.reads, list(NODES))

    def test_changed_resolution_cannot_reuse_same_host_schedule(self):
        backend = NativeScheduleReadback()
        wrong_clock = prepare_pulse_protocol(
            dict(thalamic_input=True, thalamic_input_type="pulses", th_pulse_start=3.0,
                 th_interval=7.0, th_delay_pulse_generator=1.0),
            dict(sim_resolution=0.2, t_presim=0.0, t_sim=20.0),
            dict(initial_emission_ms=[2.0, 9.0, 10.0, 11.0, 14.0], decision_time_ms=10.0))
        with mock.patch.object(control.ctypes, "CDLL", side_effect=AssertionError("mutation loader called")) as loader:
            with self.assertRaises(ValueError):
                control.replace_masked_tc_future(backend, NODES, MASK, wrong_clock)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
