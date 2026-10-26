import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

import brian2 as b2
import numpy as np

from .pfc_factorized_source import install_uniform_projections
from .pfc_initial_rng import install_initial_voltage_rng


SOURCE_ENTRY = "neurodynex3/working_memory_network/wm_model.py"


class InitialVoltageRngTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_root = Path(os.environ.get(
            "NEURONSIM_PFC_SOURCE",
            str(Path(__file__).resolve().parents[1] / "external" / "pfc_working_memory")))
        cls.entry = cls.source_root / SOURCE_ENTRY
        cls.path_added = str(cls.source_root) not in sys.path
        if cls.path_added:
            sys.path.insert(0, str(cls.source_root))

    @classmethod
    def tearDownClass(cls):
        if cls.path_added:
            sys.path.remove(str(cls.source_root))

    def setUp(self):
        b2.start_scope()
        b2.prefs.codegen.target = "numpy"

    def load_source(self):
        spec = importlib.util.spec_from_file_location("pfc_initial_rng_test_source", self.entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def assert_rng_state_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertEqual(left[2:], right[2:])

    def draw_pair(self, module):
        inhibitory = module.numpy.random.uniform(-60.0, high=-50.0, size=4)
        excitatory = module.numpy.random.uniform(-60.0, high=-50.0, size=16)
        return np.concatenate((excitatory, inhibitory))

    def run_native(self, module, initial_seed, noise_seed):
        b2.start_scope()
        b2.seed(noise_seed)
        trace = install_initial_voltage_rng(module, initial_seed)
        results = module.simulate_wm(
            N_excitatory=16, N_inhibitory=4, monitored_subset_size=16,
            sim_time=2 * b2.ms)
        excitatory, inhibitory = results[2], results[6]
        np.testing.assert_array_equal(
            np.asarray(inhibitory.v[:, 0] / b2.mV),
            trace.draws[0].values_mV[np.asarray(inhibitory.record)])
        np.testing.assert_array_equal(
            np.asarray(excitatory.v[:, 0] / b2.mV),
            trace.draws[1].values_mV[np.asarray(excitatory.record)])
        noise = np.concatenate((
            np.asarray(excitatory.source.s_AMPA[:]),
            np.asarray(inhibitory.source.s_AMPA[:])))
        return trace, noise, np.random.get_state()

    def test_global_numpy_namespace_and_state_unchanged(self):
        module = self.load_source()
        global_uniform = np.random.uniform
        np.random.seed(314)
        before = np.random.get_state()
        trace = install_initial_voltage_rng(module, 41)
        actual = self.draw_pair(module)
        self.assertIs(np.random.uniform, global_uniform)
        self.assert_rng_state_equal(before, np.random.get_state())
        self.assertIs(module.numpy.multiply, np.multiply)
        self.assertIs(module.numpy.linspace, np.linspace)
        expected_rng = np.random.RandomState(41)
        expected_i = expected_rng.uniform(-60.0, -50.0, 4)
        expected_e = expected_rng.uniform(-60.0, -50.0, 16)
        np.testing.assert_array_equal(actual, np.concatenate((expected_e, expected_i)))
        np.testing.assert_array_equal(trace.initial_voltage_mV, actual)

    def test_reinstallation_and_new_modules_keep_independent_traces(self):
        first, second = self.load_source(), self.load_source()
        trace_a = install_initial_voltage_rng(first, 41)
        a = self.draw_pair(first)
        self.assertIs(second.numpy, np)
        trace_b = install_initial_voltage_rng(second, 42)
        b = self.draw_pair(second)
        trace_c = install_initial_voltage_rng(first, 41)
        c = self.draw_pair(first)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(trace_a.initial_voltage_mV, a)
        np.testing.assert_array_equal(trace_b.initial_voltage_mV, b)
        self.assertFalse(np.array_equal(a, b))
        self.assertIsNot(trace_a, trace_c)
        self.assertIsNot(first.numpy, second.numpy)
        self.assertFalse(trace_a.draws[0].values_mV.flags.writeable)
        c[:] = 0
        np.testing.assert_array_equal(trace_c.initial_voltage_mV, a)

    def test_native_source_order_units_and_brian_noise_stream(self):
        module = self.load_source()
        a, noise_a, state_a = self.run_native(module, 41, 314)
        b, noise_b, state_b = self.run_native(module, 42, 314)
        c, noise_c, _ = self.run_native(module, 41, 315)
        self.assertEqual([draw.population for draw in a.draws], ["I", "E"])
        self.assertEqual([draw.values_mV.shape for draw in a.draws], [(4,), (16,)])
        self.assertTrue(a.complete)
        self.assertTrue(all(draw.units == "mV" for draw in a.draws))
        self.assertTrue(all(draw.unit_order == "ascending_zero_based_population_index" for draw in a.draws))
        self.assertFalse(np.array_equal(a.initial_voltage_mV, b.initial_voltage_mV))
        np.testing.assert_array_equal(a.initial_voltage_mV, c.initial_voltage_mV)
        np.testing.assert_array_equal(noise_a, noise_b)
        self.assert_rng_state_equal(state_a, state_b)
        self.assertTrue(np.any(noise_a > 0))
        self.assertFalse(np.array_equal(noise_a, noise_c))

    def test_works_with_uniform_projection_adapter(self):
        module = self.load_source()
        factory = install_uniform_projections(module)
        trace = install_initial_voltage_rng(module, 41)
        self.assertIs(module.Synapses, factory)
        self.draw_pair(module)
        self.assertTrue(trace.complete)
        install_uniform_projections(module)
        self.assertIs(module.numpy.random, trace)

    def test_rejects_incomplete_extra_and_unsupported_draws(self):
        module = self.load_source()
        trace = install_initial_voltage_rng(module, 41)
        with self.assertRaisesRegex(ValueError, "TRACE_INCOMPLETE"):
            _ = trace.initial_voltage_mV
        for size in (0, -1, True, 4.0, (4,), np.array([4])):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "SHAPE_UNSUPPORTED"):
                module.numpy.random.uniform(-60.0, high=-50.0, size=size)
        for low, high in ((-61.0, -50.0), (-60.0, -49.0), (np.nan, -50.0), ([-60.0], -50.0)):
            with self.subTest(low=low, high=high), self.assertRaisesRegex(ValueError, "BOUNDS_UNSUPPORTED"):
                module.numpy.random.uniform(low, high=high, size=4)
        self.assertEqual(trace.draws, ())
        self.draw_pair(module)
        with self.assertRaisesRegex(ValueError, "DRAW_ORDER_UNSUPPORTED"):
            module.numpy.random.uniform(-60.0, high=-50.0, size=4)

    def test_rejects_invalid_seed_and_unsupported_source(self):
        module = self.load_source()
        for seed in (None, True, -1, 2**32, 41.0, np.int64(41)):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "SEED_INVALID"):
                install_initial_voltage_rng(module, seed)
        self.assertIs(module.numpy, np)
        with tempfile.TemporaryDirectory() as temporary:
            modified = Path(temporary) / "wm_model.py"
            modified.write_bytes(self.entry.read_bytes() + b"\n# modified\n")
            module.__file__ = str(modified)
            with self.assertRaisesRegex(ValueError, "REQUIRES_PINNED_SOURCE"):
                install_initial_voltage_rng(module, 41)
        fake = ModuleType("not_the_source")
        fake.__file__ = str(self.entry)
        fake.simulate_wm = lambda: None
        fake.numpy = np
        with self.assertRaisesRegex(ValueError, "SOURCE_FUNCTION_UNSUPPORTED"):
            install_initial_voltage_rng(fake, 41)
        module = self.load_source()
        module.numpy = object()
        with self.assertRaisesRegex(ValueError, "NUMPY_NAMESPACE_UNSUPPORTED"):
            install_initial_voltage_rng(module, 41)


if __name__ == "__main__":
    unittest.main()
