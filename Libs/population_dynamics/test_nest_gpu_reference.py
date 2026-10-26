"""Numerical oracle and clock/error regression tests; no native simulator needed."""
import math
import unittest
import json
from pathlib import Path
import tempfile

import yaml
from decimal import Decimal, localcontext

from Libs.population_dynamics.nest_gpu_reference import (
    align_trace, psc_voltage_factor, trace_errors, make_cases, cpu_prearrival_trace,
    load_protocol, SOURCE_REVISION, verify_completed_suite,
)


def decimal_factor(h_ms, tau_m_ms, tau_s_ms, capacitance_pf):
    with localcontext() as context:
        context.prec = 80
        h, tm, ts, capacitance = (
            Decimal.from_float(float(x))
            for x in (h_ms, tau_m_ms, tau_s_ms, capacitance_pf)
        )
        a = (-h / tm).exp()
        if tm == ts:
            return float(h * a / capacitance)
        return float((a - (-h / ts).exp())
                     / (capacitance * (Decimal(1) / ts - Decimal(1) / tm)))


class PropagatorTests(unittest.TestCase):
    def test_equal_limit(self):
        for h, tau, c in [(0.1, 10., 250.), (1., 2., 100.), (.01, 20., 500.)]:
            self.assertTrue(math.isclose(psc_voltage_factor(h, tau, tau, c),
                                        h / c * math.exp(-h / tau), rel_tol=2e-13))

    def test_near_equal_and_unequal_independent_decimal_oracle(self):
        tm = 10.
        for ts in (math.nextafter(tm, 0.), math.nextafter(tm, math.inf),
                   tm * (1 - 1e-8), tm * (1 + 1e-8),
                   tm * (1 - 1e-4), tm * (1 + 1e-4), .5, 2., 20.):
            with self.subTest(tau_s=ts):
                actual = psc_voltage_factor(.1, tm, ts, 250.)
                self.assertGreater(actual, 0.)
                self.assertTrue(math.isclose(actual, decimal_factor(.1, tm, ts, 250.),
                                             rel_tol=2e-13))

    def test_invalid_parameters(self):
        for x in (0., -1., math.inf, math.nan):
            with self.assertRaises(ValueError):
                psc_voltage_factor(.1, 10., x, 250.)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.ref = [[0., -70., 0., 0.], [.1, -69., 10., -5.], [.2, -68., 20., -10.]]

    def test_roundoff_maps_to_exact_ticks(self):
        trace = [[t, -70., 1., -1.] for t in (.1, .2, .1 + .2)]
        self.assertEqual(set(align_trace(trace, .1, 1e-9)), {1, 2, 3})

    def test_non_increasing_and_duplicate_ticks_are_rejected(self):
        for times in ((.1, .1), (.2, .1), (.1, .1 + 5e-10)):
            with self.assertRaises(ValueError):
                align_trace([[t, -70., 0., 0.] for t in times], .1, 1e-9)

    def test_half_step_is_not_snapped(self):
        with self.assertRaises(ValueError):
            align_trace([[.15, -70., 1., -1.]], .1, 1e-9)

    def test_nonfinite_values_rejected(self):
        for column in range(4):
            for invalid in (math.nan, math.inf, -math.inf):
                row = [.1, -70., 1., -1.]
                row[column] = invalid
                with self.assertRaises(ValueError):
                    align_trace([row], .1, 1e-9)

    def test_identical_trace(self):
        result = trace_errors(self.ref, self.ref, .1, 1e-9)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["rmse_mv"], 0.)

    def test_constant_bias_is_not_removed(self):
        for bias in (-2.5, 2.5):
            actual = [[t, v + bias, ex, inh] for t, v, ex, inh in self.ref]
            result = trace_errors(self.ref, actual, .1, 1e-9)
            self.assertEqual(result["max_abs_mv"], abs(bias))
            self.assertEqual(result["rmse_mv"], abs(bias))

    def test_rms_and_both_current_channels(self):
        actual = [[t, v + dv, ex + de, inh + di] for (t, v, ex, inh), dv, de, di
                  in zip(self.ref, (1., -2., 3.), (1., -4., 0.), (-5., 0., 2.))]
        result = trace_errors(self.ref, actual, .1, 1e-9)
        self.assertEqual(result["max_abs_mv"], 3.)
        self.assertEqual(result["max_abs_current_pa"], 5.)
        self.assertAlmostEqual(result["rmse_mv"], math.sqrt(14 / 3))

    def test_full_tick_shift_uses_physical_times(self):
        actual = [[t + .1, v, ex, inh] for t, v, ex, inh in self.ref]
        result = trace_errors(self.ref, actual, .1, 1e-9)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["max_abs_mv"], 1.)
        self.assertEqual(result["max_abs_current_pa"], 10.)

    def test_missing_interior_sample_rejected(self):
        with self.assertRaises(ValueError):
            trace_errors(self.ref, [self.ref[0], self.ref[2]], .1, 1e-9)

    def test_near_equal_cases_remain_distinct(self):
        cases = make_cases({"time_steps_ms": [.1, .025]})
        self.assertEqual(len(cases), 42)
        self.assertEqual(len(set(c["name"] for c in cases)), 42)


class RecorderPhaseTests(unittest.TestCase):
    def test_only_actual_arriving_currents_are_removed(self):
        result = {
            "case": {"dt_ms": .1, "delay_ms": .2,
                     "ex_weight_pa": 10., "in_weight_pa": -3.},
            "source_spike_times_ms": [[.1], [.1]],
            "trace": [[.2, -65., 5., -1.], [.3, -64., 15., -4.], [.4, -63., 12., -3.]],
        }
        actual = cpu_prearrival_trace(result)
        self.assertEqual(actual, [[.2, -65., 5., -1.], [.3, -64., 5., -1.], [.4, -63., 12., -3.]])
        self.assertEqual(result["trace"][1], [.3, -64., 15., -4.])

    def test_does_not_invent_requested_but_unemitted_spikes(self):
        result = {
            "case": {"dt_ms": .1, "delay_ms": .2,
                     "ex_weight_pa": 10., "in_weight_pa": -3.,
                     "ex_times_ms": [.1]},
            "source_spike_times_ms": [[], []],
            "trace": [[.3, -64., 5., -1.]],
        }
        self.assertEqual(cpu_prearrival_trace(result), result["trace"])

    def test_unknown_input_projection_is_rejected(self):
        with self.assertRaises(ValueError):
            cpu_prearrival_trace({"case": {"dt_ms": .1}, "source_spike_times_ms": [[]]})


class ProtocolCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(campaign="neural_population_dynamics", run_class="engineering_diagnostic",
                        source_revision=SOURCE_REVISION, time_steps_ms=[.1, .025],
                        voltage_atol_mv=.0005, current_atol_pa=.005, time_atol_ms=.00001)

    def test_missing_or_duplicate_time_steps_cannot_make_a_vacuous_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.yaml"
            for clocks in ([], [.1], [.025], [.1, .1]):
                path.write_text(yaml.safe_dump(dict(self.cfg, time_steps_ms=clocks)))
                with self.assertRaisesRegex(ValueError, "both predeclared"):
                    load_protocol(path)

    def test_changed_threshold_and_unfinished_suite_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "resolved_protocol.json").write_text(json.dumps(self.cfg))
            receipt = dict(status="RUNNING", backend="gpu",
                           cases=[{"name": c["name"], "exit_code": 0}
                                  for c in make_cases(self.cfg)])
            (directory / "execution.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "executed protocol"):
                verify_completed_suite(directory, dict(self.cfg, voltage_atol_mv=10.), "gpu")
            with self.assertRaisesRegex(ValueError, "complete successful"):
                verify_completed_suite(directory, self.cfg, "gpu")
            receipt["status"] = "EXECUTED"
            (directory / "execution.json").write_text(json.dumps(receipt))
            verify_completed_suite(directory, self.cfg, "gpu")


if __name__ == "__main__":
    unittest.main()
