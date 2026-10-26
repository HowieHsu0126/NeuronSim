"""Native input observation tests; execute on a compute node with Brian 2.5.4."""
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import brian2 as b2
import numpy as np

from .pfc_input_observation import NativeInputObserver, _DrawRecorder, _observe_sampler


class InputSamplerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.clock = SimpleNamespace(t=0 * b2.second, dt=0.1 * b2.ms)
        self.path = Path(self.temporary.name) / "draws.incomplete.npy"
        self.recorders = []
        self.addCleanup(lambda: [recorder.close() for recorder in self.recorders])

    def recorder(self, ticks=3):
        recorder = _DrawRecorder(self.path, 4, ticks, 0.0001, self.clock)
        self.recorders.append(recorder)
        return recorder

    def test_native_rng_draws_argument_and_return_identity(self):
        np.random.seed(93)
        expected = [np.random.binomial(1000, 0.00014, size=4) for _ in range(3)]
        expected_state = np.random.get_state()
        np.random.seed(93)
        calls, returns = [], []
        def original(argument):
            calls.append(argument)
            value = np.random.binomial(1000, 0.00014, size=4)
            returns.append(value)
            return value
        recorder = self.recorder()
        observed = _observe_sampler(original, recorder)
        indices = (4, np.arange(4, dtype=np.int32), np.int64(4))
        for tick, argument in enumerate(indices):
            self.clock.t = tick * 0.1 * b2.ms
            result = observed(argument)
            self.assertIs(result, returns[-1])
            self.assertIs(calls[-1], argument)
            np.testing.assert_array_equal(result, expected[tick])
        recorder.validate_complete()
        actual_state = np.random.get_state()
        self.assertEqual(expected_state[0], actual_state[0])
        np.testing.assert_array_equal(expected_state[1], actual_state[1])
        self.assertEqual(expected_state[2:], actual_state[2:])
        recorder.close()
        np.testing.assert_array_equal(np.load(self.path, allow_pickle=False), expected)
        self.assertEqual(len(calls), 3)

    def test_incomplete_duplicate_and_skipped_ticks_rejected(self):
        recorder = self.recorder()
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            recorder.validate_complete()
        recorder.record(np.array([0, 1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "CLOCK_OR_DRAW_ORDER"):
            recorder.record(np.zeros(4, dtype=np.int64))
        self.clock.t = 0.2 * b2.ms
        with self.assertRaisesRegex(ValueError, "CLOCK_OR_DRAW_ORDER"):
            recorder.record(np.zeros(4, dtype=np.int64))
        recorder.close()
        self.assertTrue(self.path.exists())
        self.assertFalse((self.path.parent / "draws.npy").exists())

    def test_invalid_native_draws_and_changed_clock_rejected(self):
        recorder = self.recorder()
        for values in (np.array([-1, 0, 0, 0]), np.array([1001, 0, 0, 0]),
                       np.zeros(4, dtype=float), np.zeros((4, 1), dtype=int), np.zeros(4, dtype=bool)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "DRAWS_INVALID"):
                recorder.record(values)
        self.assertEqual(recorder.next_tick, 0)
        self.clock.dt = 0.2 * b2.ms
        with self.assertRaisesRegex(ValueError, "CLOCK_OR_DRAW_ORDER"):
            recorder.record(np.zeros(4, dtype=np.int64))

    def test_partial_reordered_indices_rejected_before_sampling(self):
        recorder = self.recorder()
        calls = []
        observed = _observe_sampler(lambda argument: calls.append(argument), recorder)
        for argument in (True, 3, 4.0, np.arange(4)[::-1], np.array([0, 1, 1, 3]), np.arange(3)):
            with self.subTest(argument=argument), self.assertRaisesRegex(ValueError, "VECTORIZATION_ORDER"):
                observed(argument)
        self.assertEqual(calls, [])


class NativeInputObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_root = Path(os.environ.get("NEURONSIM_PFC_SOURCE", str(
            Path(__file__).resolve().parents[1] / "external" / "pfc_working_memory")))
        cls.entry = cls.source_root / "neurodynex3/working_memory_network/wm_model.py"
        cls.path_added = str(cls.source_root) not in sys.path
        if cls.path_added:
            sys.path.insert(0, str(cls.source_root))

    @classmethod
    def tearDownClass(cls):
        if cls.path_added:
            sys.path.remove(str(cls.source_root))

    def load_source(self):
        spec = importlib.util.spec_from_file_location("pfc_input_observation_test_source", self.entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_source(self, path, mode):
        module = self.load_source()
        observer = NativeInputObserver(path, mode=mode)
        original = module.PoissonInput
        observer.install(module)
        case = dict(case_id="native", source_rng_seed=231)
        config = dict(excitatory_neurons=16, inhibitory_neurons=4, dt_ms=0.1, duration_ms=2.0)
        try:
            observer.begin_case(case, config)
            b2.start_scope()
            b2.defaultclock.dt = 0.1 * b2.ms
            b2.prefs.codegen.target = "numpy"
            random.seed(231)
            b2.seed(231)
            result = module.simulate_wm(N_excitatory=16, N_inhibitory=4,
                                        monitored_subset_size=16, sim_time=2 * b2.ms)
            observer.finish_case(case, result)
            report = observer.finalize()
        finally:
            observer.close()
        self.assertIs(module.PoissonInput, original)
        return report

    def test_native_capture_and_untouched_audit_have_identical_terminal_state_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = self.run_source(root / "capture", "capture")
            audit = self.run_source(root / "audit", "audit")
            for key in ("terminal_state", "numpy_rng"):
                with np.load(root / "capture" / capture["cases"][0][key + "_file"], allow_pickle=False) as left:
                    with np.load(root / "audit" / audit["cases"][0][key + "_file"], allow_pickle=False) as right:
                        self.assertEqual(set(left.files), set(right.files))
                        for name in left.files:
                            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
            left = json.loads((root / "capture" / capture["cases"][0]["python_rng_file"]).read_text())
            right = json.loads((root / "audit" / audit["cases"][0]["python_rng_file"]).read_text())
            self.assertEqual(left, right)
            for mode in ("capture", "audit"):
                receipt = json.loads((root / mode / "native/observation.json").read_text())
                self.assertEqual([row["population"] for row in receipt["providers"]], ["I", "E"])
                self.assertFalse(receipt["gpu_parity_qualified"])
                self.assertFalse(receipt["terminal_snapshot_is_complete_restart_state"])
                for provider in receipt["providers"]:
                    if mode == "capture":
                        counts = np.load(root / mode / provider["counts_file"], allow_pickle=False)
                        self.assertEqual(counts.dtype, np.uint16)
                        self.assertEqual(counts.shape, (20, provider["units"]))
                        self.assertEqual(provider["observed_sampler_calls"], 20)
                        self.assertEqual(int(counts.sum()), provider["captured_external_events"])
                    else:
                        self.assertIsNone(provider["observed_sampler_calls"])
                        self.assertNotIn("counts_file", provider)
                codes = json.loads((root / mode / "native/generated_code.json").read_text())
                if mode == "capture":
                    self.assertTrue(codes["native_network_schedule"])
                    functions = {row["function_name"] for row in codes["native_network_schedule"]}
                    self.assertIn("update_nmda_sum", functions)
                    self.assertIn("stimulate_network", functions)
                else:
                    self.assertIsNone(codes["native_network_schedule"])
                for population in ("I", "E"):
                    self.assertEqual({row["kind"] for row in codes["populations"][population]},
                                     {"state_updater", "threshold:spike", "reset:spike"})
                    self.assertTrue(all(row["generated_code"] for row in codes["populations"][population]))

    def test_close_restores_factory_and_incomplete_case_cannot_finalize(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.load_source()
            original = module.PoissonInput
            observer = NativeInputObserver(Path(temporary) / "capture")
            observer.install(module)
            observer.begin_case({"case_id": "incomplete"}, dict(
                excitatory_neurons=16, inhibitory_neurons=4, dt_ms=0.1, duration_ms=2.0))
            with self.assertRaisesRegex(ValueError, "NOT_COMPLETE"):
                observer.finalize()
            observer.close()
            observer.close()
            self.assertIs(module.PoissonInput, original)
            self.assertFalse((observer.output / "report.json").exists())
            self.assertFalse((observer.output / "incomplete/observation.json").exists())

    def test_source_factory_restored_when_capture_flush_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.load_source()
            original = module.PoissonInput
            observer = NativeInputObserver(Path(temporary) / "capture")
            observer.install(module)
            recorder = Mock()
            recorder.close.side_effect = OSError("fixture write failure")
            observer._providers = [{"recorder": recorder}]
            with self.assertRaisesRegex(OSError, "fixture write failure"):
                observer.close()
            self.assertIs(module.PoissonInput, original)
            self.assertTrue(observer._closed)

    def test_normal_approximation_and_changed_backend_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.load_source()
            observer = NativeInputObserver(Path(temporary) / "capture")
            observer.install(module)
            observer.begin_case({"case_id": "normal"}, dict(
                excitatory_neurons=16, inhibitory_neurons=4, dt_ms=5.0, duration_ms=10.0))
            try:
                b2.start_scope()
                b2.defaultclock.dt = 5 * b2.ms
                b2.prefs.codegen.target = "numpy"
                group = b2.NeuronGroup(4, "s_AMPA:1")
                with self.assertRaisesRegex(ValueError, "NORMAL_APPROXIMATION"):
                    module.PoissonInput(group, "s_AMPA", N=1000, rate=1.4 * b2.Hz, weight=1.0)
                b2.prefs.codegen.target = "auto"
                with self.assertRaisesRegex(ValueError, "2_5_4_NUMPY"):
                    module.PoissonInput(group, "s_AMPA", N=1000, rate=1.4 * b2.Hz, weight=1.0)
            finally:
                b2.prefs.codegen.target = "numpy"
                observer.close()


if __name__ == "__main__":
    unittest.main()
