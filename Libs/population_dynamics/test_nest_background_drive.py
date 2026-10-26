"""Positive and falsifying controls for source-background reconstruction."""
import unittest
import numpy as np
from .nest_background_drive import arrival_intervals, decode_background, known_arrivals, validate_spike_counts
from .nest_reference import EDGE_DTYPE


def native_current(counts, weights, known, decay, initial=None):
    current=np.zeros(counts.shape[1],dtype=np.float32) if initial is None else np.asarray(initial,dtype=np.float32)
    rows=[current.astype(float)]
    for c,k in zip(counts,known):
        # Source combines weights in FP64, then casts the current update to FP32.
        impulse=np.asarray(weights*c,dtype=np.float32).astype(float)+k
        current=np.asarray(current.astype(float)+impulse,dtype=np.float32)
        current=np.asarray(current*decay,dtype=np.float32)
        rows.append(current.astype(float))
    return np.asarray(rows)


class BackgroundIntervalsTests(unittest.TestCase):
    def test_exact_recovery_with_recurrent_inputs_and_multiplicity(self):
        rng=np.random.default_rng(78001)
        counts=rng.poisson(.8,size=(200,3)).astype(np.int64)
        counts[8]=[0,2,3]
        weights=np.asarray([878.0849352920843,1.25,11.3],dtype=np.float32).astype(float)
        known=rng.integers(0,3,size=counts.shape)*np.asarray([345.12,1.7,20.5],dtype=np.float32).astype(float)
        number=(known>0).astype(np.int64)
        decay=np.asarray([.8187307,.9,.3],dtype=np.float32).astype(float)
        current=native_current(counts,weights,known,decay)
        recovered,diagnostics=decode_background(current,decay,known,number,weights)
        np.testing.assert_array_equal(recovered,counts)
        self.assertLess(diagnostics["max_count_interval_width"],.01)

    def test_zero_unknown_inhibitory_input_contained_by_interval(self):
        known=np.asarray([[0.],[3512.34],[0.],[8000.],[0.]])
        known=known.astype(np.float32).astype(float)
        n=(known>0).astype(np.int64);a=np.asarray([.8],dtype=np.float32).astype(float)
        current=native_current(np.zeros_like(n),np.ones(1),known,a)
        lo,hi=arrival_intervals(current,a,known,n)
        self.assertTrue(np.all(lo<=0));self.assertTrue(np.all(hi>=0))
        shifted=np.roll(known,1,axis=0)
        lo,hi=arrival_intervals(current,a,shifted,np.roll(n,1,axis=0))
        self.assertTrue(np.any(lo>0)|np.any(hi<0))

    def test_fractional_event_rejected_instead_of_rounded(self):
        counts=np.asarray([[.25],[0.]])
        a=np.asarray([.8],dtype=np.float32).astype(float);known=np.zeros_like(counts)
        current=native_current(counts,np.asarray([1.]),known,a)
        with self.assertRaisesRegex(ValueError,"noninteger or ambiguous"):
            decode_background(current,a,known,np.zeros_like(counts,dtype=np.int64),np.ones(1))

    def test_ambiguous_interval_rejected(self):
        current=np.asarray([[2.**22],[2.**22]])
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            decode_background(current,np.ones(1),np.zeros((1,1)),np.zeros((1,1),dtype=np.int64),np.ones(1))

    def test_invalid_states_and_weights_rejected(self):
        for kind in ["negative","nonfinite","non_fp32","weight_zero","weight_float64"]:
            c=np.zeros((3,1));a=np.asarray([.8],dtype=np.float32).astype(float)
            w=np.ones(1)
            if kind=="negative":c[0,0]=-1
            elif kind=="nonfinite":c[0,0]=np.nan
            elif kind=="non_fp32":c[0,0]=.1
            elif kind=="weight_zero":w[0]=0
            else:w[0]=.1
            with self.assertRaises(ValueError):decode_background(c,a,np.zeros((2,1)),np.zeros((2,1),dtype=np.int64),w)

    def test_subnormal_propagator_rejected_before_false_zero_event_decode(self):
        with self.assertRaisesRegex(ValueError,"normal propagators"):
            decode_background(np.zeros((2,1)),np.asarray([2.**-127]),np.zeros((1,1)),
                              np.zeros((1,1),dtype=np.int64),np.asarray([4.]))

    def test_native_counters_detect_omitted_spike(self):
        evidence={"unit_ids":np.asarray([11,23]),"native_counts":np.asarray([1,0]),
                  "recorded_counts":np.asarray([1,0])}
        validate_spike_counts(np.asarray([11,23]),evidence,np.asarray([11]))
        with self.assertRaisesRegex(ValueError,"incomplete"):
            validate_spike_counts(np.asarray([11,23]),evidence,np.asarray([],dtype=np.int64))

    def test_parallel_signed_edges_and_delays_preserved(self):
        edges=np.asarray([(2,1,1.25,.1),(2,1,2.5,.1),(2,1,-4.,.2)],dtype=EDGE_DTYPE)
        arrivals,number=known_arrivals(edges,np.asarray([1]),np.asarray([1]),np.asarray([2]),5,.1)
        self.assertEqual(arrivals[2,0,0],3.75);self.assertEqual(number[2,0,0],2)
        self.assertEqual(arrivals[3,0,1],4.);self.assertEqual(number[3,0,1],1)
        self.assertEqual(arrivals.sum(),7.75)

    def test_duplicate_spikes_and_missing_target_rejected(self):
        edges=np.asarray([(2,1,1.,.1)],dtype=EDGE_DTYPE)
        with self.assertRaisesRegex(ValueError,"more than once"):
            known_arrivals(edges,np.asarray([1,1]),np.asarray([1,1]),np.asarray([2]),5,.1)
        with self.assertRaisesRegex(ValueError,"not in recorded"):
            known_arrivals(edges,np.asarray([1]),np.asarray([1]),np.asarray([1]),5,.1)


if __name__=="__main__":unittest.main()
