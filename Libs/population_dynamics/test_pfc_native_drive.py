"""Validate source draw order, state ownership, and numerical stream integration."""
import unittest
import numpy as np
from .pfc_native_drive import create_drive
from .pfc_deterministic_replay import simulate


def setting(policy='source_global'):
    c=dict(excitatory_neurons=8,inhibitory_neurons=4,dt_ms=0.05,duration_ms=1.0,
           weight_scaling_factor=2.0,random_stream_policy=policy,stimulus_start_ms=100.0,
           stimulus_duration_ms=100.0,stimulus_strength_nA=0.0)
    row=dict(case_id='native_stream_test',stimulus_strength_nA=0.0,distractor_strength_nA=0.0)
    if policy=='source_global':c['source_rng_seed']=71
    else:c.update(initial_rng_seed=71,external_noise_seed=72)
    return c,row


def same_rng(test,left,right):
    for a,b in zip(left,right):np.testing.assert_array_equal(a,b)


class NativeDriveTests(unittest.TestCase):
    def test_global_stream_matches_literal_source_calls_and_final_rng(self):
        c,row=setting();initial,drive=create_drive(c,row)
        original=np.random.RandomState(71)
        init_i=original.uniform(-60.,high=-50.,size=4)
        init_e=original.uniform(-60.,high=-50.,size=8)
        np.testing.assert_array_equal(initial,np.concatenate((init_e,init_i)))
        target={'I':[],'E':[]}
        for _ in range(20):
            for pop,size in (('I',4),('E',8)):
                target[pop].append(original.binomial(1000,1.4*(0.05/1000),size=size))
        parts=[drive.take(rows) for rows in (1,7,12)]
        for pop in ('I','E'):
            actual=np.concatenate([p[pop] for p in parts])
            np.testing.assert_array_equal(actual,np.asarray(target[pop],dtype=np.uint16))
            self.assertEqual(actual.dtype,np.uint16)
            self.assertEqual(drive.event_totals[pop],int(actual.sum()))
        same_rng(self,drive.get_state(),original.get_state())
        self.assertEqual(drive.ticks,20)

    def test_independent_external_stream_starts_before_any_voltage_draw(self):
        c,row=setting('independent_initial_external');initial,drive=create_drive(c,row)
        original=np.random.RandomState(72);block=drive.take(20)
        expected_i=[];expected_e=[]
        for _ in range(20):
            expected_i.append(original.binomial(1000,1.4*(0.05/1000),4))
            expected_e.append(original.binomial(1000,1.4*(0.05/1000),8))
        np.testing.assert_array_equal(block['I'],expected_i)
        np.testing.assert_array_equal(block['E'],expected_e)
        same_rng(self,drive.get_state(),original.get_state())

    def test_invalid_requests_do_not_consume_rng_or_clock(self):
        c,row=setting();_,drive=create_drive(c,row);state=drive.get_state()
        for rows in (0,-1,True,1.5,21):
            with self.assertRaises(ValueError):drive.take(rows)
            same_rng(self,drive.get_state(),state)
            self.assertEqual(drive.ticks,0)

    def test_returned_chunks_remain_independent_after_later_take(self):
        c,row=setting();_,drive=create_drive(c,row)
        first=drive.take(10);saved={k:v.copy() for k,v in first.items()};drive.take(10)
        for pop in ('I','E'):np.testing.assert_array_equal(first[pop],saved[pop])

    def test_streamed_solver_matches_identical_materialized_source_inputs(self):
        for policy in ('source_global','independent_initial_external'):
            c,row=setting(policy);initial,materialized=create_drive(c,row);values=materialized.take(20)
            reference=simulate(c,row,initial,values['E'],values['I'])
            initial,streamed=create_drive(c,row);candidate=simulate(c,row,initial,drive=streamed)
            for family in ('raw','terminal_state'):
                for key in reference[family]:np.testing.assert_array_equal(reference[family][key],candidate[family][key])
            self.assertEqual(candidate['inputs_used']['input_mode'],'native_numpy_stream')
            for pop in ('I','E'):self.assertEqual(candidate['inputs_used'][pop]['events'],int(values[pop].sum()))

    def test_solver_rejects_reusing_an_advanced_source_stream(self):
        c,row=setting();initial,drive=create_drive(c,row);drive.take(1)
        with self.assertRaises(ValueError):simulate(c,row,initial,drive=drive)
        self.assertEqual(drive.ticks,1)


if __name__=='__main__':unittest.main()
