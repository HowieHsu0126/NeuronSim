"""Physical and scheduling invariants independent of saved reference outputs."""
import unittest
import hashlib
import math
from scipy.special import erf
from unittest import mock
import numpy as np
from . import pfc_deterministic_replay as replay
from .pfc_deterministic_replay import simulate
from .pfc_native_drive import create_drive
from .pfc_kernel_parameters import resolve_kernel_parameters, circuit_provenance
from .provenance import canonical_json


def inputs(duration_ms=0.05):
    config=dict(excitatory_neurons=8,inhibitory_neurons=4,weight_scaling_factor=2.0,
                dt_ms=0.05,duration_ms=duration_ms,stimulus_center_deg=0.0,stimulus_width_deg=90.0,
                stimulus_start_ms=100.0,stimulus_duration_ms=100.0,stimulus_strength_nA=0.0,
                distractor_center_deg=180.0,distractor_width_deg=90.0,distractor_start_ms=400.0,
                distractor_duration_ms=50.0)
    case=dict(case_id='source_invariant',stimulus_strength_nA=0.0,distractor_strength_nA=0.0)
    ticks=round(duration_ms/0.05)
    return config,case,np.full(12,-55.0,dtype=np.float64),np.zeros((ticks,8),np.uint16),np.zeros((ticks,4),np.uint16)


class DeterministicPFCInvariants(unittest.TestCase):
    def test_passive_membrane_matches_analytic_midpoint_update(self):
        c,k,v,e,i=inputs()
        r=simulate(c,k,v,e,i)
        for pop,decay in (('E',50.0),('I',100.0)):
            step=0.00005*decay
            expected=-0.070+0.015*(1-step+step*step/2)
            np.testing.assert_allclose(r['terminal_state'][pop+'__v'],expected,rtol=0,atol=1e-16)
        self.assertEqual(r['raw']['spike_ticks'].size,0)
        self.assertEqual(r['inputs_used']['ticks_consumed'],1)

    def test_source_current_wraparound_and_distractor_persistence(self):
        c,k,v,e,i=inputs(2.5)
        c.update(stimulus_start_ms=0.0,stimulus_duration_ms=10.0,distractor_start_ms=0.0,distractor_duration_ms=1.0)
        k.update(stimulus_strength_nA=0.06,distractor_strength_nA=0.03)
        r=simulate(c,k,v,e,i)
        expected=np.zeros(8);expected[[7,0,1]]=0.06*1e-9;expected[[3,4,5]]=0.03*1e-9
        np.testing.assert_array_equal(r['terminal_state']['E__I_stim'],expected)

    def test_refractory_voltage_frozen_while_synaptic_states_advance(self):
        c,k,v,e,i=inputs(1.0)
        c.update(stimulus_start_ms=0.0,stimulus_duration_ms=10.0,stimulus_width_deg=360.0)
        k['stimulus_strength_nA']=1.0;v[:8]=-50.0001;e[0]=100
        r=simulate(c,k,v,e,i);state=r['terminal_state']
        np.testing.assert_array_equal(r['raw']['native_spike_counts'][:8],np.ones(8,dtype=np.int64))
        np.testing.assert_array_equal(state['E__v'],np.full(8,-60.0*0.001))
        np.testing.assert_array_equal(state['E__lastspike'],np.zeros(8))
        self.assertFalse(state['E__not_refractory'].any())
        for field,tau,initial in (('x',1.88,1.0),('s_AMPA',1.8,100.0)):
            h=0.05/tau
            np.testing.assert_allclose(state['E__'+field],initial*(1-h+h*h/2)**19,rtol=2e-14,atol=1e-14)
        self.assertTrue((state['E__s_NMDA']>0).all())

    def test_inhibitory_self_exclusion_and_post_threshold_external_input(self):
        c,k,v,e,i=inputs(0.1);i[0]=1000
        r=simulate(c,k,v,e,i);state=r['terminal_state']
        np.testing.assert_array_equal(r['raw']['spike_ticks'],np.ones(4,dtype=np.int64))
        np.testing.assert_array_equal(r['raw']['spike_unit_indices'],np.arange(8,12))
        np.testing.assert_array_equal(state['E__s_GABA'],np.full(8,4.0))
        np.testing.assert_array_equal(state['I__s_GABA'],np.full(4,3.0))

    def test_replay_consumes_no_numpy_randomness_and_preserves_inputs(self):
        c,k,v,e,i=inputs(0.1);e[0,2]=4;i[0,1]=3
        before=np.random.get_state();inputs_before=[a.copy() for a in (v,e,i)]
        first=simulate(c,k,v,e,i);second=simulate(c,k,v,e,i);after=np.random.get_state()
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        for a,b in zip((v,e,i),inputs_before):np.testing.assert_array_equal(a,b)
        for name in first['raw']:np.testing.assert_array_equal(first['raw'][name],second['raw'][name])
        self.assertTrue(first['inputs_used']['no_resampling'])

    def test_observed_case_current_uses_native_clock_and_enters_rk2(self):
        config, case, initial, counts_e, counts_i = inputs(4.05)
        config.update(stimulus_start_ms=1.25, stimulus_duration_ms=1.5, stimulus_strength_nA=0.06)
        case["stimulus_strength_nA"] = 0.5
        consumed = []
        original_rk2 = replay._rk2

        def record_consumed_current(state, constants, t, dt, arithmetic, population):
            if population == "E":
                consumed.append(state["I_stim"].copy())
            return original_rk2(state, constants, t, dt, arithmetic, population)

        with mock.patch.object(replay, "_rk2", side_effect=record_consumed_current):
            result = simulate(config, case, initial, counts_e, counts_i, observe_applied_current=True)
        observation = result["applied_current_observation"]
        arrays = observation["arrays"]
        # Source control runs at 1 ms: a 1.25--2.75 ms request is applied at 2--3 ms.
        np.testing.assert_array_equal(arrays["segment_start_ticks"], [0, 40, 60])
        np.testing.assert_array_equal(arrays["segment_stop_ticks"], [40, 60, 81])
        expected = np.zeros((3, 8))
        expected[1, [7, 0, 1]] = 0.5 * 1e-9
        np.testing.assert_array_equal(arrays["current_A"], expected)
        np.testing.assert_array_equal(arrays["source_unit_indices"], np.arange(8))
        np.testing.assert_array_equal(arrays["archive_unit_indices"], np.arange(8))
        for start, stop, current in zip(arrays["segment_start_ticks"], arrays["segment_stop_ticks"], expected):
            np.testing.assert_array_equal(np.asarray(consumed)[start:stop], np.broadcast_to(current, (stop - start, 8)))
        self.assertEqual(observation["metadata"]["units"], "A")
        self.assertEqual(observation["metadata"]["timing"], "after_source_stimulus_operation_before_rk2")

    def test_observation_preserves_overlap_and_native_residual_current(self):
        config, case, initial, counts_e, counts_i = inputs(5.05)
        config.update(stimulus_start_ms=0.0, stimulus_duration_ms=4.0,
                      distractor_start_ms=1.0, distractor_duration_ms=1.0,
                      distractor_center_deg=45.0, distractor_width_deg=90.0)
        case.update(stimulus_strength_nA=0.5, distractor_strength_nA=0.25)
        result = simulate(config, case, initial, counts_e, counts_i, observe_applied_current=True)
        arrays = result["applied_current_observation"]["arrays"]
        np.testing.assert_array_equal(arrays["segment_start_ticks"], [0, 20, 40, 80])
        np.testing.assert_array_equal(arrays["segment_stop_ticks"], [20, 40, 80, 101])
        np.testing.assert_array_equal(arrays["cue_active"], [True, True, True, False])
        np.testing.assert_array_equal(arrays["distractor_active"], [False, True, False, False])
        expected = np.zeros((4, 8))
        expected[:3, [7, 0, 1]] = 0.5 * 1e-9
        expected[1, [0, 1, 2]] = 0.25 * 1e-9
        expected[2, 2] = 0.25 * 1e-9  # Outside the cue, the native distractor residual persists.
        np.testing.assert_array_equal(arrays["current_A"], expected)
        np.testing.assert_array_equal(result["terminal_state"]["E__I_stim"], expected[-1])

    def test_current_snapshots_read_backend_state_and_survive_later_mutation(self):
        config, case, initial, counts_e, counts_i = inputs(2.05)
        config.update(stimulus_start_ms=1.0, stimulus_duration_ms=1.0)
        case["stimulus_strength_nA"] = 0.5
        original_held_current = replay._held_current
        marker = 0.123 * 1e-9

        def add_backend_marker(current, program, operation_time):
            active = original_held_current(current, program, operation_time)
            if operation_time == 0.0:
                current[2] = marker
            return active

        with mock.patch.object(replay, "_held_current", side_effect=add_backend_marker):
            result = simulate(config, case, initial, counts_e, counts_i, observe_applied_current=True)
        current = result["applied_current_observation"]["arrays"]["current_A"]
        self.assertEqual(current[0, 2], marker)
        self.assertEqual(current[1, 2], marker)
        np.testing.assert_array_equal(current[-1], np.zeros(8))
        self.assertFalse(np.shares_memory(current, result["terminal_state"]["E__I_stim"]))
        current[0, 2] = 99.0
        self.assertEqual(current[1, 2], marker)
        np.testing.assert_array_equal(result["terminal_state"]["E__I_stim"], np.zeros(8))

    def test_current_observation_leaves_streamed_outputs_and_rng_bitwise_unchanged(self):
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
                plain = simulate(config, case, initial, drive=plain_drive)
                observed_initial, observed_drive = create_drive(config, case)
                observed = simulate(config, case, observed_initial, drive=observed_drive,
                                    observe_applied_current=True)
                np.testing.assert_array_equal(initial, observed_initial)
                for family in ("raw", "terminal_state"):
                    for key in plain[family]:
                        left, right = plain[family][key], observed[family][key]
                        self.assertEqual(left.dtype, right.dtype)
                        self.assertEqual(left.shape, right.shape)
                        self.assertEqual(left.tobytes(), right.tobytes(), (policy, family, key))
                for left, right in zip(plain_drive.get_state(), observed_drive.get_state()):
                    np.testing.assert_array_equal(left, right)
                self.assertEqual(plain["inputs_used"], observed["inputs_used"])
                self.assertNotIn("applied_current_observation", plain)
                self.assertEqual(plain["cost"]["applied_current_capture_seconds"], 0.0)
                self.assertGreaterEqual(observed["cost"]["applied_current_capture_seconds"], 0.0)
                self.assertLessEqual(observed["cost"]["applied_current_capture_seconds"], observed["cost"]["wall_seconds"])
        for left, right in zip(global_before, np.random.get_state()):
            np.testing.assert_array_equal(left, right)

    def test_wrong_input_length_or_noninteger_counts_are_rejected(self):
        c,k,v,e,i=inputs()
        for invalid in (np.zeros((2,8),np.uint16),e.astype(np.float64)):
            with self.assertRaises(ValueError):simulate(c,k,v,invalid,i)

    def test_out_of_binomial_support_rejected(self):
        c,k,v,e,i=inputs();e[0,0]=1001
        with self.assertRaises(ValueError):simulate(c,k,v,e,i)



class SourceKernelParameterTests(unittest.TestCase):
    def test_default_and_explicit_default_retain_original_scalar_arithmetic(self):
        for n_e in (8, 128):
            sigma, j_pos = 20.0, 1.6
            tmp = math.sqrt(2. * math.pi) * sigma * erf(180. / math.sqrt(2.) / sigma) / 360.
            j_neg = (1. - j_pos * tmp) / (1. - tmp)
            original = np.asarray([j_neg + (j_pos - j_neg) * math.exp(
                -.5 * (360. * min(j, n_e - j) / n_e) ** 2 / sigma ** 2)
                for j in range(n_e)], dtype=np.float64)
            self.assertEqual(replay._kernel(n_e).tobytes(), original.tobytes())
            self.assertEqual(replay._kernel(n_e, dict(sigma_weight_profile=20,
                Jpos_excit2excit=1.6)).tobytes(), original.tobytes())

    def test_parameters_change_real_recurrent_state_and_preserve_default_outputs(self):
        config, case, initial, counts_e, counts_i = inputs(1.)
        counts_e[0, 0] = 1000
        first = simulate(config, case, initial, counts_e, counts_i)
        explicit = simulate(dict(config, sigma_weight_profile=20., Jpos_excit2excit=1.6),
                            case, initial, counts_e, counts_i)
        changed = simulate(dict(config, sigma_weight_profile=30., Jpos_excit2excit=1.7),
                           case, initial, counts_e, counts_i)
        self.assertGreater(first['raw']['native_spike_counts'].sum(), 0)
        for family in ('raw', 'terminal_state'):
            for key in first[family]:
                self.assertEqual(first[family][key].tobytes(), explicit[family][key].tobytes())
        self.assertFalse(np.array_equal(first['terminal_state']['E__s_NMDA_total'],
                                        changed['terminal_state']['E__s_NMDA_total']))
        self.assertFalse(np.array_equal(replay._kernel(16), replay._kernel(16, dict(sigma_weight_profile=30.))))
        self.assertFalse(np.array_equal(replay._kernel(16), replay._kernel(16, dict(Jpos_excit2excit=1.7))))
        self.assertEqual(first['inputs_used'], changed['inputs_used'])

    def test_nonfinite_singular_or_negative_weights_are_rejected(self):
        invalid = [dict(sigma_weight_profile=value) for value in
                   (True, 0., -1., float('nan'), float('inf'), 1e-200, 1e200)]
        invalid += [dict(Jpos_excit2excit=value) for value in
                    (True, -1., float('nan'), float('inf'), 100.)]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, 'PFC_SOURCE_KERNEL_'):
                replay._kernel(16, parameters)
        self.assertTrue(np.all(replay._kernel(16, dict(Jpos_excit2excit=0.)) >= 0.))

    def test_default_identity_compatibility_and_weighted_connectivity_identity(self):
        config, _, _, _, _ = inputs()
        source = replay.SOURCE_ENTRY_SHA256
        intrinsic = {key: config[key] for key in (
            'excitatory_neurons', 'inhibitory_neurons', 'weight_scaling_factor', 'dt_ms')}
        topology = dict(excitatory_kernel='source_circulant_gaussian', inhibitory_self_connections=False,
                        ei_and_ie='source_all_to_all', neurons=[8, 4], source_entry_sha256=source)
        legacy = dict(parameter_identity=hashlib.sha256(canonical_json(
            dict(intrinsic, source_entry_sha256=source)).encode()).hexdigest(),
            connectivity_identity=hashlib.sha256(canonical_json(topology).encode()).hexdigest())
        self.assertEqual(circuit_provenance(config, source), legacy)
        self.assertEqual(circuit_provenance(dict(config, sigma_weight_profile=20, Jpos_excit2excit=1.6), source), legacy)
        for changes in (dict(sigma_weight_profile=30.), dict(Jpos_excit2excit=1.7)):
            different = circuit_provenance(dict(config, **changes), source)
            for key in legacy:
                self.assertNotEqual(different[key], legacy[key])
        clock = circuit_provenance(dict(config, dt_ms=.025), source)
        self.assertEqual(clock['connectivity_identity'], legacy['connectivity_identity'])
        self.assertNotEqual(clock['parameter_identity'], legacy['parameter_identity'])
        self.assertEqual(circuit_provenance(dict(config, source_rng_seed=123), source), legacy)


if __name__=='__main__':unittest.main()
