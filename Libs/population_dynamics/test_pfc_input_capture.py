"""Source-binding tests for native input observation comparison."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .pfc_input_capture import _bind_native_case
from .provenance import sha256_file


class NativeObservationBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "native").mkdir()
        (self.root / "inputs").mkdir()
        self.case = dict(case_id="case", source_rng_seed=23, stimulus_strength_nA=0.07,
                         distractor_strength_nA=0.0)
        self.config = dict(excitatory_neurons=2, inhibitory_neurons=1, dt_ms=0.05, duration_ms=0.5)
        self.report = dict(resolved_config=self.config, source_entry_sha256="source")
        self.arrays = dict(E__source_indices=np.arange(2), I__source_indices=np.arange(1),
                           E__archive_indices=np.arange(2), I__archive_indices=np.array([2]),
                           E__spike_counts=np.array([1, 0]), I__spike_counts=np.array([0]),
                           E__v=np.array([-.060, -.065]), I__v=np.array([-.064]))
        raw = self.root / "native/raw.npz"
        np.savez(raw, native_spike_counts=np.array([1, 0, 0]), spike_unit_indices=np.array([0]),
                 terminal_voltage_mV=np.concatenate([self.arrays["E__v"], self.arrays["I__v"]]) / .001)
        self.native = dict(self.case, core_units=3, native_events=1, raw_file="raw.npz", raw_sha256=sha256_file(raw))
        self.row = dict(case_id="case")
        for name in ("terminal_state", "numpy_rng", "python_rng", "generated_code"):
            self.row[name + "_file"] = name + ".npz"
            self.row[name + "_sha256"] = "not_consumed_by_this_helper"
        dt = .05 / 1000
        providers = [dict(population=population, units=n, expected_sample_rows=10, dt_seconds=dt,
                          terminal_clock_seconds=10 * dt, target_variable="s_AMPA", N=1000,
                          rate_Hz=1.4, probability=1.4 * dt, weight_dimensionless=1., when="synapses", order=0,
                          source_unit_order="ascending_zero_based_population_index", counts_shape=[10, n],
                          observed_sampler_calls=10) for population, n in (("I", 1), ("E", 2))]
        self.receipt = dict(self.row, mode="capture", case=self.case.copy(), source_entry_sha256="source",
                            providers=providers)
        self.save_snapshot()

    def save_snapshot(self):
        path = self.root / "inputs/terminal_state.npz"
        np.savez(path, **self.arrays)
        self.row["terminal_state_sha256"] = sha256_file(path)
        self.receipt["terminal_state_sha256"] = self.row["terminal_state_sha256"]

    def validate(self):
        return _bind_native_case(self.root, self.row, self.receipt, self.report, self.native, self.case)

    def test_snapshot_binds_to_native_counts_voltages_and_clock(self):
        self.assertEqual(self.validate(), 6)

    def test_self_consistent_stale_snapshot_rejected(self):
        for key in ("E__spike_counts", "I__v"):
            with self.subTest(key=key):
                previous = self.arrays[key].copy()
                self.arrays[key] = previous + 1
                self.save_snapshot()
                with self.assertRaisesRegex(ValueError, "NATIVE_STATE_MISMATCH"):
                    self.validate()
                self.arrays[key] = previous
                self.save_snapshot()

    def test_capture_geometry_cannot_override_native_config(self):
        self.receipt["providers"][0].update(expected_sample_rows=9, counts_shape=[9, 1], observed_sampler_calls=9)
        with self.assertRaisesRegex(ValueError, "NATIVE_INPUT_GEOMETRY_MISMATCH"):
            self.validate()

    def test_case_and_identity_mismatch_rejected(self):
        self.receipt["case"]["source_rng_seed"] = 24
        with self.assertRaisesRegex(ValueError, "NATIVE_CASE_MISMATCH"):
            self.validate()
        self.receipt["case"] = self.case.copy()
        self.arrays["E__archive_indices"] = np.array([1, 0])
        self.save_snapshot()
        with self.assertRaisesRegex(ValueError, "NATIVE_UNIT_MAPPING_MISMATCH"):
            self.validate()


if __name__ == "__main__": unittest.main()
