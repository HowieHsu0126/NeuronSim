import unittest
import numpy as np
from .pfc_response_analysis import summarize_counts

class PopulationVectorTests(unittest.TestCase):
    def test_quadrants_and_spike_weighting(self):
        counts=np.zeros(10,dtype=int);counts[2]=3;counts[8]=2
        x=summarize_counts(counts,8,90.,30.,200.)
        self.assertAlmostEqual(x['decoded_angle_deg'],90.)
        self.assertAlmostEqual(x['resultant_length'],1.)
        self.assertEqual(x['excitatory_rate_hz'],3/8/.2)
        self.assertEqual(x['inhibitory_rate_hz'],2/2/.2)
    def test_silent_and_cancelled_vectors_are_undefined(self):
        for counts in [np.zeros(10,dtype=int),np.array([1,0,0,0,1,0,0,0,0,0])]:
            x=summarize_counts(counts,8,350.,30.,100.)
            self.assertFalse(x['direction_defined'])
            self.assertIsNone(x['decoded_angle_deg'])
    def test_wraparound_and_periodic_cue_region(self):
        counts=np.zeros(10,dtype=int);counts[0]=1
        x=summarize_counts(counts,8,350.,90.,100.)
        self.assertAlmostEqual(x['circular_cue_error_deg'],10.)
        self.assertEqual(x['cue_units'],3)
        self.assertGreater(x['cue_selectivity_hz'],0)
    def test_invalid_counts_and_exposure_fail(self):
        for count,duration in [(np.array([-1,2,0]),100.),(np.array([.5,1.,0.]),100.),(np.array([1,0,0]),0.)]:
            with self.assertRaises(ValueError):summarize_counts(count,2,0.,30.,duration)

if __name__=='__main__':unittest.main()
