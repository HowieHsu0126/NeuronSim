"""Focused deterministic IEEE arithmetic-model and source-DAG checks."""
from fractions import Fraction as F
import unittest

import numpy as np

from population_dynamics import pfc_synaptic_roundoff as r
from population_dynamics.pfc_synaptic_trace_bounds import NmdaParameters, nmda_rk2_step
from population_dynamics.pfc_deterministic_replay import _Arithmetic, _constants, _rk2


def source_parameters(dt_ms):
    constants = _constants({"weight_scaling_factor": 1.0}, "E")
    dt = dt_ms * 1e-3
    parameters = NmdaParameters(F.from_float(dt), F.from_float(constants["tau_nmda_x"]),
                               F.from_float(constants["tau_nmda_s"]), F.from_float(constants["alpha_nmda"]))
    refractory = int((constants["refractory"] + 1e-3 * dt) / dt)
    return parameters, constants, refractory


class SynapticRoundoffTests(unittest.TestCase):
    def test_conservative_primitive_pairs(self):
        a = r.ErrorPair(r.Interval(F(-1), F(2)), F(1, 16))
        b = r.ErrorPair(r.Interval(F(1), F(3)), F(1, 32))
        for operation, actual_op in ((r.add, lambda x, y: x+y), (r.subtract, lambda x, y: x-y),
                                      (r.multiply, lambda x, y: x*y), (r.divide_positive, lambda x, y: x/y)):
            result = operation(a, b)
            for x in (F(-1), F(1, 2), F(2)):
                for y in (F(1), F(2), F(3)):
                    nominal = actual_op(x, y)
                    self.assertLessEqual(result.nominal.lower, nominal)
                    self.assertLessEqual(nominal, result.nominal.upper)
                    for dx in (-a.error, a.error):
                        for dy in (-b.error, b.error):
                            actual = F.from_float(actual_op(float(x+dx), float(y+dy)))
                            self.assertLessEqual(abs(actual-nominal), result.error)

    def test_division_and_finite_operation_failures(self):
        with self.assertRaisesRegex(ValueError, "denominator"):
            r.divide_positive(r.constant(F(1)), r.ErrorPair(r.Interval(F(1), F(2)), F(1)))
        with self.assertRaisesRegex(ValueError, "overflow"):
            r.multiply(r.constant(r.MAX_FINITE), r.constant(F(2)))
        with self.assertRaises(ValueError):
            r.Interval(F(1), F(0))
        with self.assertRaises(ValueError):
            r.ErrorPair(r.Interval(F(0), F(1)), F(-1))

    def test_subnormal_primitive_enclosures(self):
        tiny = F(1, 2**1074)
        for a, b in ((tiny, F(1, 2)), (F(1, 2**1025), F(1, 8))):
            product = r.multiply(r.constant(a), r.constant(b))
            actual = F.from_float(float(a)*float(b))
            self.assertLessEqual(abs(actual-a*b), product.error)

    def test_source_dag_against_exact_targets_at_all_clocks(self):
        for dt_ms in (0.05, 0.025, 0.0125):
            parameters, constants, _ = source_parameters(dt_ms)
            dt = float(parameters.dt_seconds)
            for s, x in ((0.0, 0.0), (0.5, 0.3), (1.0, 1.3),
                         (float(F(1,2**1074)), float(F(1,2**1074))),
                         (float(F(1,2**1025)), float(F(1,2**1025)))):
                local = r.nmda_local_roundoff(parameters,
                    s_box=r.Interval(F.from_float(s), F.from_float(s)),
                    x_box=r.Interval(F.from_float(x), F.from_float(x)))
                state = {name: np.zeros(1, dtype=np.float64) for name in
                         ("s_AMPA","s_GABA","s_NMDA_total","I_stim")}
                state.update(v=np.asarray([-0.06], dtype=np.float64),
                             lastspike=np.asarray([-1e4], dtype=np.float64),
                             x=np.asarray([x], dtype=np.float64), s_NMDA=np.asarray([s], dtype=np.float64))
                _rk2(state, constants, 0.0, dt, _Arithmetic("numpy"), "E")
                target_s, target_x = nmda_rk2_step(F.from_float(s), F.from_float(x), parameters)
                self.assertEqual(local.s.nominal.lower, target_s)
                self.assertEqual(local.s.nominal.upper, target_s)
                self.assertEqual(local.x_before_spike.nominal.lower, target_x)
                self.assertLessEqual(abs(F.from_float(float(state["s_NMDA"][0]))-target_s), local.s.error)
                self.assertLessEqual(abs(F.from_float(float(state["x"][0]))-target_x), local.x_before_spike.error)
                state["x"] += 1.0
                self.assertLessEqual(abs(F.from_float(float(state["x"][0]))-(target_x+1)), local.x_after_spike.error)

    def test_source_expanded_boxes_and_uniform_radii(self):
        for dt_ms in (0.05, 0.025, 0.0125):
            parameters, _, refractory = source_parameters(dt_ms)
            bound = r.same_event_history_bound(parameters, refractory_ticks=refractory,
                                               x_radius=F(1,2**40), s_radius=F(1,2**30))
            self.assertGreaterEqual(bound.x_slack, 0)
            self.assertGreaterEqual(bound.s_slack, 0)
            self.assertLess(bound.derivatives.q_s, 1)
            self.assertLessEqual(bound.local.x_after_spike.error, F(1,2**50))
            self.assertLessEqual(bound.local.s.error, F(1,2**51))
            self.assertEqual(bound.initial_x_upper_bound, 0)
            self.assertEqual(bound.initial_s_box, r.Interval(F(0), F(0)))
            self.assertEqual(bound.initial_x_error_bound, 0)
            self.assertEqual(bound.initial_s_error_bound, 0)
            self.assertFalse(bound.native_backend_certified)
            self.assertFalse(bound.source_confidence_certified)

    def test_derivative_rectangle_bound_and_exact_difference(self):
        parameters, _, refractory = source_parameters(0.05)
        bound = r.same_event_history_bound(parameters, refractory_ticks=refractory,
                                           x_radius=F(1,2**40), s_radius=F(1,2**30))
        e = bound.derivatives
        h, alpha = parameters.dt_seconds, parameters.alpha_per_second
        a0, d = 1/parameters.tau_s_seconds, 1-h/(2*parameters.tau_x_seconds)
        def phi(s, x):
            a, am = a0+alpha*x, a0+alpha*d*x
            return (1-h*am*(1-h*a/2))*s+h*alpha*x*(d-h*am/2)
        step = F(1,2**60)
        for x in (e.x_box.lower, (e.x_box.lower+e.x_box.upper)/2, e.x_box.upper):
            for s in (e.s_box.lower, F(1,2), e.s_box.upper):
                derivative = (phi(s,x+step)-phi(s,x-step))/(2*step)
                self.assertEqual(derivative, r.derivative_x(parameters,x,s))
                self.assertLessEqual(abs(derivative), e.l_x)
                ds = (phi(s+step,x)-phi(s-step,x))/(2*step)
                self.assertGreaterEqual(ds, 0)
                self.assertLessEqual(ds, e.q_s)

    def test_too_small_radii_and_initial_error_are_rejected(self):
        parameters, _, refractory = source_parameters(0.05)
        for dx, ds in ((F(1,2**80), F(1,2**30)), (F(1,2**40), F(1,2**80))):
            with self.assertRaises(ValueError):
                r.same_event_history_bound(parameters, refractory_ticks=refractory,
                                           x_radius=dx, s_radius=ds)
        with self.assertRaisesRegex(ValueError, "initial errors"):
            r.same_event_history_bound(parameters, refractory_ticks=refractory,
                x_radius=F(1,2**40), s_radius=F(1,2**30), initial_x_error_bound=F(1,2**39))
        with self.assertRaisesRegex(ValueError, "initial s"):
            r.same_event_history_bound(parameters, refractory_ticks=refractory,
                x_radius=F(1,2**40), s_radius=F(1,2**30), initial_s_box=r.Interval(F(-1),F(0)))

    def test_bad_source_constants_or_expanded_premises_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary64"):
            r.nmda_local_roundoff(NmdaParameters(F(1,3), F(1), F(1), F(1)),
                                  s_box=r.Interval(F(0),F(1)), x_box=r.Interval(F(0),F(1)))
        parameters, _, _ = source_parameters(0.05)
        with self.assertRaises(ValueError):
            r.expanded_derivative_bounds(parameters,x_ceiling=F(2),x_radius=F(1),s_radius=F(1))
        with self.assertRaises(ValueError):
            r.expanded_derivative_bounds(parameters,x_ceiling=F(2),x_radius=F(1,2**40),s_radius=F(-1))

    def test_horizon_bound_is_constant_cost_and_includes_initial_state(self):
        parameters, _, refractory = source_parameters(0.05)
        bound = r.same_event_history_bound(parameters,refractory_ticks=refractory,
                                           x_radius=F(1,2**40),s_radius=F(1,2**30))
        x_l1, s_l1 = r.history_l1_error_bound(bound,400000)
        e_x, e_s = r.propagated_error_envelope(bound)
        self.assertEqual(x_l1,parameters.dt_seconds*400000*e_x)
        self.assertEqual(s_l1,parameters.dt_seconds*400000*e_s)
        self.assertLess(e_x, bound.x_radius)
        self.assertLess(e_s, bound.s_radius)
        with self.assertRaises(ValueError):
            r.history_l1_error_bound(bound,True)


    def test_propagated_envelope_retains_nonzero_initial_errors(self):
        parameters, _, refractory = source_parameters(0.05)
        initial_x, initial_s = F(1,2**42), F(1,2**32)
        bound = r.same_event_history_bound(parameters,refractory_ticks=refractory,
            x_radius=F(1,2**40),s_radius=F(1,2**30),
            initial_x_error_bound=initial_x,initial_s_error_bound=initial_s)
        e_x, e_s = r.propagated_error_envelope(bound)
        self.assertGreaterEqual(e_x, initial_x)
        self.assertGreaterEqual(e_s, initial_s)
        self.assertLessEqual(e_x, bound.x_radius)
        self.assertLessEqual(e_s, bound.s_radius)
        x_l1, s_l1 = r.history_l1_error_bound(bound,1)
        self.assertGreaterEqual(x_l1, parameters.dt_seconds*initial_x)
        self.assertGreaterEqual(s_l1, parameters.dt_seconds*initial_s)

    def test_short_source_trajectory_obeys_sharp_same_event_envelope(self):
        parameters, constants, refractory = source_parameters(0.05)
        h = parameters.dt_seconds
        horizon, events = refractory+3, {0, refractory}
        for initial_x, initial_s in ((F(0),F(0)),(F(1,2**42),F(1,2**32))):
            bound = r.same_event_history_bound(parameters,refractory_ticks=refractory,
                x_radius=F(1,2**40),s_radius=F(1,2**30),
                initial_x_error_bound=initial_x,initial_s_error_bound=initial_s)
            e_x, e_s = r.propagated_error_envelope(bound)
            state = {name:np.zeros(1,dtype=np.float64) for name in
                     ("s_AMPA","s_GABA","s_NMDA_total","I_stim")}
            state.update(v=np.asarray([-0.06],dtype=np.float64),
                         lastspike=np.asarray([-1e4],dtype=np.float64),
                         x=np.asarray([float(initial_x)],dtype=np.float64),
                         s_NMDA=np.asarray([float(initial_s)],dtype=np.float64))
            reference_s, reference_x = F(0), F(0)
            errors_x, errors_s = [], []
            arithmetic = _Arithmetic("numpy")
            for tick in range(horizon+1):
                dx = abs(F.from_float(float(state["x"][0]))-reference_x)
                ds = abs(F.from_float(float(state["s_NMDA"][0]))-reference_s)
                self.assertLessEqual(dx,e_x)
                self.assertLessEqual(ds,e_s)
                errors_x.append(dx);errors_s.append(ds)
                if tick==horizon:
                    break
                _rk2(state,constants,tick*float(h),float(h),arithmetic,"E")
                reference_s, reference_x = nmda_rk2_step(reference_s,reference_x,parameters)
                if tick in events:
                    reference_x += 1
                    state["x"] += 1.0
                    state["v"][:] = constants["reset"]
                    state["lastspike"][:] = tick*float(h)
            x_l1,s_l1 = r.history_l1_error_bound(bound,horizon)
            for selected in (slice(None,-1),slice(1,None)):
                self.assertLessEqual(h*sum(errors_x[selected],F(0)),x_l1)
                self.assertLessEqual(h*sum(errors_s[selected],F(0)),s_l1)


if __name__ == "__main__":
    unittest.main()
