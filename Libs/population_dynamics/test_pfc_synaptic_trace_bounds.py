"""Deterministic unittest contracts for exact-real synaptic trace bounds."""
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import numpy as np

from population_dynamics import pfc_synaptic_trace_bounds as bounds
from population_dynamics.pfc_deterministic_replay import _Arithmetic, _constants, _rk2
from population_dynamics.pfc_kernel_parameters import source_kernel


def small_parameters():
    return bounds.NmdaParameters(F(1, 64), F(1, 8), F(1, 2), F(1))


def exact_history(history, parameters, horizon):
    x, s = history.initial_x, history.initial_s
    events, states_x, states_s = set(history.spike_ticks), [x], [s]
    for tick in range(horizon):
        s, x = bounds.nmda_rk2_step(s, x, parameters)
        x += int(tick in events)
        states_x.append(x)
        states_s.append(s)
    return states_x, states_s


class SynapticTraceBoundsTests(unittest.TestCase):
    def test_refractory_ceiling_with_explicit_initial_contribution(self):
        q, refractory = F(3, 4), 3
        for initial in (F(0), F(1, 4)):
            ceiling = bounds.refractory_x_ceiling(q, refractory, initial_x_upper_bound=initial)
            events = bounds.validate_spike_ticks((0, 3, 6, 9), refractory_ticks=refractory, horizon_ticks=16)
            x = initial
            for tick in range(16):
                x = q * x + int(tick in events)
                self.assertLessEqual(x, ceiling)
        self.assertEqual(bounds.refractory_x_ceiling(q, 3), F(64, 37))

    def test_source_constants_and_midpoint_state_update_compatibility(self):
        constants = _constants({"weight_scaling_factor": 1.0}, "E")
        arithmetic = _Arithmetic("numpy")
        for dt_ms, refractory in ((0.05, 40), (0.025, 80), (0.0125, 160)):
            dt = dt_ms * 1e-3
            parameters = bounds.NmdaParameters(F.from_float(dt),
                F.from_float(constants["tau_nmda_x"]), F.from_float(constants["tau_nmda_s"]),
                F.from_float(constants["alpha_nmda"]))
            native_refractory = int((constants["refractory"] + 1e-3 * dt) / dt)
            self.assertEqual(native_refractory, refractory)
            ceiling = bounds.refractory_x_ceiling(bounds.rk2_decay_factor(
                dt_seconds=parameters.dt_seconds, tau_seconds=parameters.tau_x_seconds), refractory)
            proof = bounds.nmda_invariance_bounds(parameters, x_upper_bound=ceiling)
            self.assertGreaterEqual(proof.q_s, 0)
            self.assertLess(proof.q_s, 1)
            for s in (0.0, 0.5, 1.0):
                for x in (0.0, 0.3, 1.3):
                    state = {name: np.zeros(1) for name in ("s_AMPA", "s_GABA", "s_NMDA_total", "I_stim")}
                    state.update(v=np.asarray([-0.06]), lastspike=np.asarray([-1e4]),
                                 x=np.asarray([x]), s_NMDA=np.asarray([s]))
                    _rk2(state, constants, 0.0, dt, arithmetic, "E")
                    expected_s, expected_x = bounds.nmda_rk2_step(F.from_float(s), F.from_float(x), parameters)
                    # Numerical source compatibility is not a native-rounding proof.
                    self.assertAlmostEqual(state["s_NMDA"][0], float(expected_s), delta=1e-15)
                    self.assertAlmostEqual(state["x"][0], float(expected_x), delta=1e-15)

    def test_invariance_and_lipschitz_for_checked_domain(self):
        parameters = small_parameters()
        proof = bounds.nmda_invariance_bounds(parameters, x_upper_bound=F(5))
        points = [(s, x) for s in (F(0), F(1, 2), F(1)) for x in (F(0), F(5, 2), F(5))]
        values = {}
        for s, x in points:
            value, _ = bounds.nmda_rk2_step(s, x, parameters)
            values[s, x] = value
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        for s, x in points:
            for other_s, other_x in points:
                self.assertLessEqual(abs(values[s, x] - values[other_s, other_x]),
                    proof.q_s * abs(s - other_s) + proof.l_x * abs(x - other_x))

    def test_exact_pulse_distance_against_finite_sum_and_tail(self):
        q, h = F(3, 4), F(1, 1000)
        for shift in (0, 1, 4, 12):
            end = shift + 5
            finite = sum((abs(q ** tick - (q ** (tick - shift) if tick >= shift else 0))
                          for tick in range(end + 1)), F(0))
            tail = q ** (end + 1 - shift) * (1 - q ** shift) / (1 - q)
            expected = h * (finite + tail)
            for sign in (-1, 1):
                self.assertEqual(bounds.pulse_translation_l1_exact(dt_seconds=h, decay_factor=q,
                                                                  shift_ticks=sign * shift), expected)
                self.assertGreaterEqual(bounds.pulse_translation_l1_bound(dt_seconds=h, decay_factor=q,
                                                                          shift_ticks=sign * shift), expected)

    def test_missing_extra_spikes_and_initial_trace_are_charged(self):
        h, q = F(1, 10), F(1, 2)
        result = bounds.pulse_train_l1_bound((0, 3), (1,), matched_spikes=((0, 1),),
            dt_seconds=h, decay_factor=q, refractory_ticks=2,
            initial_left=F(0), initial_right=F(1, 4), horizon_ticks=5)
        self.assertEqual(result, 2 * h + h / (1 - q) + h * F(1, 4) / (1 - q))
        self.assertEqual(bounds.pulse_train_l1_bound((), (2,), matched_spikes=(),
            dt_seconds=h, decay_factor=q, refractory_ticks=2, initial_left=F(0),
            initial_right=F(0), horizon_ticks=5), h / (1 - q))

    def test_invalid_actual_spike_sequences(self):
        for spikes, refractory, horizon in (((2, 1), 1, 4), ((0, 0), 1, 4),
                                            ((0, 1), 2, 4), ((-1,), 1, 4),
                                            ((True,), 1, 4), ((1.0,), 1, 4),
                                            ((4,), 1, 4), ((), 0, 4)):
            with self.subTest(spikes=spikes):
                with self.assertRaises(ValueError):
                    bounds.validate_spike_ticks(spikes, refractory_ticks=refractory, horizon_ticks=horizon)

    def test_invalid_matching_cannot_drop_or_reuse_events(self):
        for pairs in (((0, 1), (0, 4)), ((3, 99),), ((True, 1),)):
            with self.assertRaises(ValueError):
                bounds.pulse_train_l1_bound((0, 3), (1, 4), matched_spikes=pairs,
                    dt_seconds=F(1, 100), decay_factor=F(1, 2), refractory_ticks=2,
                    initial_left=F(0), initial_right=F(0), horizon_ticks=5)

    def test_invalid_exact_units_and_decay_conditions(self):
        for dt, tau in ((0.05, F(1)), (F(0), F(1)), (F(-1), F(1)),
                        (F(1), F(0)), (F(2), F(1))):
            with self.assertRaises(ValueError):
                bounds.rk2_decay_factor(dt_seconds=dt, tau_seconds=tau)
        for q in (F(1), F(-1), 0.5):
            with self.assertRaises(ValueError):
                bounds.refractory_x_ceiling(q, 2)
        with self.assertRaises(ValueError):
            bounds.pulse_translation_l1_bound(dt_seconds=F(1), decay_factor=F(1, 2), shift_ticks=True)

    def test_invalid_nmda_sufficient_conditions_are_errors(self):
        for parameters, x in ((bounds.NmdaParameters(F(1), F(1), F(1), F(1)), F(1)),
                               (small_parameters(), F(100)),
                               (bounds.NmdaParameters(F(1, 2), F(3, 10), F(1), F(1)), F(0))):
            with self.assertRaises(ValueError):
                bounds.nmda_invariance_bounds(parameters, x_upper_bound=x)
        with self.assertRaises(ValueError):
            bounds.SynapticHistory(0, F(1, 100), (), F(0), F(2))

    def test_nonlinear_finite_history_bound_dominates_exact_traces(self):
        parameters = small_parameters()
        left = bounds.SynapticHistory(7, parameters.dt_seconds, (0, 3, 8), F(1, 8), F(1, 4))
        right = bounds.SynapticHistory(7, parameters.dt_seconds, (1, 4), F(0), F(1, 8))
        result = bounds.nmda_history_l1_bound(left, right, matched_spikes=((0, 1), (3, 4)),
            parameters=parameters, refractory_ticks=2, initial_x_upper_bound=F(1, 8), horizon_ticks=12)
        _, left_s = exact_history(left, parameters, 12)
        _, right_s = exact_history(right, parameters, 12)
        actual = parameters.dt_seconds * sum((abs(a - b) for a, b in zip(left_s[1:], right_s[1:])), F(0))
        self.assertLessEqual(actual, result)
        self.assertLessEqual(result, 12 * parameters.dt_seconds)

    def test_equal_full_histories_have_zero_comparison_bound(self):
        parameters = small_parameters()
        history = bounds.SynapticHistory(7, parameters.dt_seconds, (0, 3, 8), F(0), F(1, 4))
        result = bounds.nmda_history_l1_bound(history, history,
            matched_spikes=((0, 0), (3, 3), (8, 8)), parameters=parameters,
            refractory_ticks=2, initial_x_upper_bound=F(0), horizon_ticks=12)
        self.assertEqual(result, 0)

    def test_history_clock_identity_and_initial_bounds_are_explicit(self):
        parameters = small_parameters()
        left = bounds.SynapticHistory(7, parameters.dt_seconds, (), F(1), F(0))
        for right, initial in ((bounds.SynapticHistory(8, parameters.dt_seconds, (), F(0), F(0)), F(1)),
                               (bounds.SynapticHistory(7, parameters.dt_seconds / 2, (), F(0), F(0)), F(1)),
                               (bounds.SynapticHistory(7, parameters.dt_seconds, (), F(0), F(0)), F(0))):
            with self.assertRaises(ValueError):
                bounds.nmda_history_l1_bound(left, right, matched_spikes=(), parameters=parameters,
                    refractory_ticks=2, initial_x_upper_bound=initial, horizon_ticks=12)

    def test_large_history_path_does_not_compute_translation_powers(self):
        parameters = small_parameters()
        left = bounds.SynapticHistory(7, parameters.dt_seconds, (0,), F(0), F(0))
        right = bounds.SynapticHistory(7, parameters.dt_seconds, (399999,), F(0), F(0))
        with patch.object(bounds, "pulse_translation_l1_exact", side_effect=AssertionError("large exponent")):
            result = bounds.nmda_history_l1_bound(left, right, matched_spikes=((0, 399999),),
                parameters=parameters, refractory_ticks=2, initial_x_upper_bound=F(0), horizon_ticks=400000)
        self.assertGreater(result, 0)

    def test_graph_triangle_bound_on_source_coupled_fixture(self):
        ids, n_e = tuple(range(6)), 4
        kernel = source_kernel(n_e)
        edges = [(source, target, F.from_float(float(kernel[(target - source) % n_e])))
                 for source in range(n_e) for target in range(n_e)]
        edges += [(source, target, F(1)) for source in range(n_e) for target in (4, 5)]
        edges += [(source, target, F(-1)) for source in (4, 5) for target in range(n_e)]
        edges += [(4, 5, F(-1)), (5, 4, F(-1))]
        columns = bounds.absolute_column_sums(ids, iter(edges))
        h = F(1, 100)
        traces = {neuron: [F(neuron - 2), F(2 - neuron), F(neuron % 2)] for neuron in ids}
        source_l1 = {neuron: h * sum(map(abs, values)) for neuron, values in traces.items()}
        result = bounds.weighted_graph_input_l1_bound(source_l1, columns)
        actual = h * sum((abs(sum((weight * traces[source][tick]
                    for source, end, weight in edges if end == target), F(0)))
                    for target in ids for tick in range(3)), F(0))
        self.assertLessEqual(actual, result)
        self.assertEqual(set(columns), set(ids))
        self.assertEqual(columns[4], F(5))  # Four E targets and the other I target; no self edge.

    def test_graph_identities_and_nonnegative_bounds_required(self):
        with self.assertRaises(ValueError):
            bounds.absolute_column_sums((0, 0), ())
        with self.assertRaises(ValueError):
            bounds.absolute_column_sums((0, 1), ((0, 2, F(1)),))
        with self.assertRaises(ValueError):
            bounds.weighted_graph_input_l1_bound({0: F(1)}, {0: F(1), 1: F(1)})
        with self.assertRaises(ValueError):
            bounds.weighted_graph_input_l1_bound({0: F(-1)}, {0: F(1)})


    def test_pre_update_interval_covers_exact_source_prefix_states(self):
        parameters = small_parameters()
        left = bounds.SynapticHistory(7, parameters.dt_seconds, (0, 3, 8), F(1, 8), F(1, 4))
        right = bounds.SynapticHistory(7, parameters.dt_seconds, (1, 4), F(0), F(1, 8))
        options = dict(matched_spikes=((0, 1), (3, 4)), parameters=parameters,
                       refractory_ticks=2, initial_x_upper_bound=F(1, 8), horizon_ticks=12)
        pre = bounds.nmda_history_l1_bound(left, right, state_interval="pre", **options)
        _, left_s = exact_history(left, parameters, 12)
        _, right_s = exact_history(right, parameters, 12)
        actual_pre = parameters.dt_seconds * sum(
            (abs(a - b) for a, b in zip(left_s[:-1], right_s[:-1])), F(0))
        self.assertLessEqual(actual_pre, pre)
        self.assertEqual(bounds.nmda_history_l1_bound(left, right, **options),
                         bounds.nmda_history_l1_bound(left, right, state_interval="post", **options))

    def test_one_tick_pre_interval_keeps_the_complete_initial_difference(self):
        parameters = small_parameters()
        left = bounds.SynapticHistory(7, parameters.dt_seconds, (), F(0), F(1))
        right = bounds.SynapticHistory(7, parameters.dt_seconds, (), F(0), F(0))
        options = dict(matched_spikes=(), parameters=parameters, refractory_ticks=2,
                       initial_x_upper_bound=F(0), horizon_ticks=1)
        pre = bounds.nmda_history_l1_bound(left, right, state_interval="pre", **options)
        post = bounds.nmda_history_l1_bound(left, right, state_interval="post", **options)
        _, left_s = exact_history(left, parameters, 1)
        _, right_s = exact_history(right, parameters, 1)
        actual_pre = parameters.dt_seconds * abs(left_s[0] - right_s[0])
        actual_post = parameters.dt_seconds * abs(left_s[1] - right_s[1])
        self.assertEqual(actual_pre, parameters.dt_seconds)
        self.assertEqual(pre, parameters.dt_seconds)
        self.assertLessEqual(actual_post, post)
        self.assertLess(post, actual_pre)

    def test_unknown_nmda_state_interval_is_rejected(self):
        parameters = small_parameters()
        history = bounds.SynapticHistory(7, parameters.dt_seconds, (), F(0), F(0))
        for selector in ("pre_update", "POST", "", None, True):
            with self.subTest(selector=selector):
                with self.assertRaisesRegex(ValueError, "state_interval"):
                    bounds.nmda_history_l1_bound(history, history, matched_spikes=(),
                        parameters=parameters, refractory_ticks=2, initial_x_upper_bound=F(0),
                        horizon_ticks=1, state_interval=selector)


if __name__ == "__main__":
    unittest.main()
