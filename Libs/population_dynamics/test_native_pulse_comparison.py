"""Verify native response comparison preserves clocks, units and signed errors."""
from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from .native_pulse_comparison import compare_candidate

N_UNITS = 100000
N_FIELDS = 128
N_BINS = 100
N_DRAWS = 16


class NativeEpisodeFixture:
    def __init__(self, ticks=None, event_units=None):
        self.dt_ms, self.bin_ms, self.bin_ticks, self.duration_ticks = 0.1, 5.0, 50, 60000
        self.unit_ids = np.arange(N_UNITS, dtype=np.int64)
        self.field_ids = np.asarray([f"field_{index}" for index in range(N_FIELDS)])
        self.unit_field_index = np.arange(N_UNITS, dtype=np.int64) % N_FIELDS
        self.minimum_interval_ticks = np.full(N_UNITS, 21, dtype=np.int64)
        self.event_ticks = np.asarray([9999, 10000, 10049, 10050, 14999, 15000] if ticks is None else ticks,
                                      dtype=np.int64)
        self.event_units = np.asarray([10, 0, 1, 0, 0, 2] if event_units is None else event_units,
                                      dtype=np.int64)
        self.source_identity = dict(dt_ms=0.1,
            gpu_source_revision="74fc021cad2b4bd5c011c390b3f95a302b663483",
            native_poisson_rng_family="per_edge_philox4_32_10", graph_manifest_sha256="a" * 64,
            native_library_sha256="b" * 64)
        self.windows = []

    def window(self, start, stop):
        self.windows.append((start, stop))
        selected = (self.event_ticks >= start) & (self.event_ticks < stop)
        result = np.zeros(((stop - start) // self.bin_ticks, N_UNITS), dtype=np.float32)
        np.add.at(result, ((self.event_ticks[selected] - start) // self.bin_ticks,
                          self.event_units[selected]), 1)
        return result, np.zeros((len(result), 1), dtype=np.float32)

    def ready_at(self, stop):
        ready = np.zeros(N_UNITS, dtype=np.int64)
        for tick, unit in zip(self.event_ticks, self.event_units):
            if tick < stop:
                ready[unit] = max(0, int(tick) + int(self.minimum_interval_ticks[unit]) - stop)
        return ready


def prediction(native, factor=1):
    totals = np.zeros((N_DRAWS, N_UNITS), dtype=np.int64)
    fields = np.zeros((N_DRAWS, N_BINS, N_FIELDS), dtype=np.int64)
    selected = (native.event_ticks >= 10000) & (native.event_ticks < 15000)
    for tick, unit in zip(native.event_ticks[selected], native.event_units[selected]):
        totals[:, unit] += factor
        fields[:, (tick - 10000) // 50, native.unit_field_index[unit]] += factor
    denominator = np.bincount(native.unit_field_index, minlength=N_FIELDS) * 0.005
    return dict(unit_ids=native.unit_ids.copy(), field_ids=native.field_ids.copy(),
                unit_total_counts=totals, field_counts=fields, field_rate_hz=fields / denominator)


def reference(native):
    return dict(field_ids=native.field_ids.copy(), field_rate_hz=np.zeros((14, N_BINS, N_FIELDS)),
                scale_hz=np.ones(N_FIELDS), block_ids=np.asarray([str(block) for block in range(14)]))


def inputs(factor=1):
    native, source = NativeEpisodeFixture(), NativeEpisodeFixture()
    # A qualified live-control patch is expected to have a different library
    # digest. This must not defeat otherwise exact causal replay comparison.
    native.source_identity["native_library_sha256"] = "c" * 64
    return native, source, prediction(native, factor), reference(native)


class NativePulseComparisonTests(unittest.TestCase):
    def test_half_open_counts_and_terminal_availability_use_actual_native_events(self):
        native, source, predicted, references = inputs()
        report, arrays = compare_candidate(native, source, predicted, references)
        self.assertEqual(native.windows, [(10000, 15000)])
        self.assertEqual(report["prefix_events_verified"], 1)
        self.assertEqual(report["actual_total_spikes"], 4)
        self.assertEqual(arrays["unit_total_counts"][0], 3)
        self.assertEqual(arrays["unit_total_counts"][1], 1)
        self.assertEqual(arrays["unit_total_counts"][2], 0)
        self.assertEqual(arrays["field_counts"][0, 0], 1)
        self.assertEqual(arrays["field_counts"][0, 1], 1)
        self.assertEqual(arrays["field_counts"][1, 0], 1)
        self.assertEqual(arrays["field_counts"][99, 0], 1)
        self.assertEqual(arrays["terminal_ready_in_ticks"][0], 20)
        self.assertEqual(arrays["terminal_ready_in_ticks"][2], 0)
        self.assertEqual(report["prediction_native_energy_score"], 0.0)
        self.assertEqual(report["unit_total_ensemble_mean_rmse_spikes"], 0.0)
        self.assertGreater(report["native_reference_standardized_rmse_mean"], 0)
        for name in ("native_future_noise_pairing_verified", "independent_intervention_qualified",
                     "inverse_intervention_qualified", "scientific_claim_qualified"):
            self.assertIs(report[name], False)

    def test_signed_underprediction_and_overprediction_are_both_retained(self):
        for factor, direction in ((0, -1), (2, 1)):
            with self.subTest(factor=factor):
                report, arrays = compare_candidate(*inputs(factor))
                self.assertEqual(np.sign(report["unit_total_ensemble_mean_bias_spikes"]), direction)
                self.assertGreater(report["unit_total_ensemble_mean_rmse_spikes"], 0)
                self.assertGreater(report["prediction_native_energy_score"], 0)
                self.assertEqual(report["status"], "DEVELOPMENT_NATIVE_PULSE_RESPONSE_COMPARED")
                self.assertEqual(int(arrays["unit_total_counts"].sum()), 4)

    def test_unit_field_map_and_clock_changes_reject_before_native_window(self):
        for alteration in ("units", "fields", "map", "graph", "dt", "bins", "duration"):
            with self.subTest(alteration=alteration):
                native, source, predicted, references = inputs()
                if alteration == "units":
                    native.unit_ids = native.unit_ids[::-1]
                elif alteration == "fields":
                    native.field_ids = native.field_ids[::-1]
                elif alteration == "map":
                    native.unit_field_index[0] = 1
                elif alteration == "graph":
                    native.source_identity["graph_manifest_sha256"] = "d" * 64
                elif alteration == "dt":
                    native.dt_ms = 0.2
                elif alteration == "bins":
                    native.bin_ticks = 25
                else:
                    native.duration_ticks = 15000
                with self.assertRaises(ValueError):
                    compare_candidate(native, source, predicted, references)
                self.assertEqual(native.windows, [])

    def test_changed_causal_event_prefix_rejects_but_future_events_may_differ(self):
        for alteration in ("tick", "unit", "removed"):
            with self.subTest(alteration=alteration):
                native, source, predicted, references = inputs()
                if alteration == "tick":
                    source.event_ticks[0] = 9998
                elif alteration == "unit":
                    source.event_units[0] = 11
                else:
                    source.event_ticks = source.event_ticks[1:]
                    source.event_units = source.event_units[1:]
                with self.assertRaises(ValueError):
                    compare_candidate(native, source, predicted, references)
                self.assertEqual(native.windows, [])
        native, source, predicted, references = inputs()
        source.event_units[1:] = 50
        report, _ = compare_candidate(native, source, predicted, references)
        self.assertTrue(report["causal_source_prefix_verified"])

    def test_saved_prediction_shapes_Hz_units_and_full_unit_totals_must_agree(self):
        for alteration in ("draws", "Hz", "totals", "units", "fields", "counts_fractional"):
            with self.subTest(alteration=alteration):
                native, source, predicted, references = inputs()
                if alteration == "draws":
                    predicted["field_counts"] = predicted["field_counts"][:15]
                elif alteration == "Hz":
                    predicted["field_rate_hz"][0, 0, 0] += 1
                elif alteration == "totals":
                    predicted["unit_total_counts"][0, 0] += 1
                elif alteration == "units":
                    predicted["unit_ids"] = predicted["unit_ids"][::-1]
                elif alteration == "fields":
                    predicted["field_ids"] = predicted["field_ids"][::-1]
                else:
                    predicted["field_counts"] = predicted["field_counts"].astype(np.float64)
                    predicted["field_counts"][0, 0, 0] += 0.5
                with self.assertRaises(ValueError):
                    compare_candidate(native, source, predicted, references)
                self.assertEqual(native.windows, [])

    def test_reference_order_scale_and_independent_block_denominator_are_required(self):
        for alteration in ("fields", "scale", "blocks", "shape"):
            with self.subTest(alteration=alteration):
                native, source, predicted, references = inputs()
                if alteration == "fields":
                    references["field_ids"] = references["field_ids"][::-1]
                elif alteration == "scale":
                    references["scale_hz"][0] = 0
                elif alteration == "blocks":
                    references["block_ids"][13] = "12"
                else:
                    references["field_rate_hz"] = references["field_rate_hz"][:13]
                with self.assertRaises(ValueError):
                    compare_candidate(native, source, predicted, references)
                self.assertEqual(native.windows, [])


if __name__ == "__main__":
    unittest.main()
