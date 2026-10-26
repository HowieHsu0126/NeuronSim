"""Paired source blocks, legal currents and first-control-difference boundaries."""
import copy
import unittest
import numpy as np
from .pfc_reference import case_settings, verify_matched_prefixes

class PFCSourcePairTests(unittest.TestCase):
    def config(self):
        return dict(source_rng_seed=91001,stimulus_strength_nA=.06,cases=[
            dict(case_id='cue',source_rng_seed=91002,distractor_strength_nA=0.),
            dict(case_id='off',source_rng_seed=91002,stimulus_strength_nA=0.,distractor_strength_nA=0.)])
    def record(self,name,seed,cue):
        return dict(case_id=name,seed=seed,ticks=np.array([1,5,11]),ids=np.array([0,1,0]),
                    control_ticks=np.array([0,10,30]),controls=np.array([[0.,0.],[cue,0.],[0.,0.]]))
    def test_explicit_blocks_and_legacy_default(self):
        rows=case_settings(self.config())
        self.assertEqual([r['source_rng_seed'] for r in rows],[91002,91002])
        self.assertEqual([r['stimulus_strength_nA'] for r in rows],[.06,0.])
        self.assertEqual(rows[0]['block_id'],'pfc_source_block_91002')
        c=self.config();del c['cases'][0]['source_rng_seed']
        self.assertEqual(case_settings(c)[0]['block_id'],'pfc_source_block_0')
    def test_invalid_case_seed_current_or_id(self):
        for k,v in [('source_rng_seed',True),('source_rng_seed',-1),('source_rng_seed',2**32),('stimulus_strength_nA',float('nan')),('stimulus_strength_nA',-.1),('case_id','../bad')]:
            c=self.config();c['cases'][0][k]=v
            with self.subTest(k=k,v=v),self.assertRaises(ValueError):case_settings(c)
        c=self.config();c['cases'][1]['case_id']='cue'
        with self.assertRaises(ValueError):case_settings(c)
    def test_prefix_stops_at_cue_and_different_rng_is_separate(self):
        a=self.record('cue',1,.06);b=self.record('off',1,0.)
        b['ticks'][-1]=15
        c=self.record('other',2,.06);c['ticks'][0]=2
        checks=verify_matched_prefixes([a,b,c],40)
        self.assertEqual(len(checks),1)
        self.assertEqual(checks[0]['prefix_stop_tick'],10)
        self.assertEqual(checks[0]['event_count'],2)
        b['ticks'][0]=2
        with self.assertRaisesRegex(ValueError,'PREFIX'):verify_matched_prefixes([a,b],40)
    def test_identical_controls_require_complete_trajectory_and_shared_clock(self):
        a=self.record('a',1,.06);b=copy.deepcopy(a);b['case_id']='b'
        self.assertEqual(verify_matched_prefixes([a,b],40)[0]['prefix_stop_tick'],40)
        b['ticks'][-1]=12
        with self.assertRaises(ValueError):verify_matched_prefixes([a,b],40)
        b=copy.deepcopy(a);b['control_ticks'][1]=11
        with self.assertRaisesRegex(ValueError,'CLOCK'):verify_matched_prefixes([a,b],40)


    def test_explicit_initial_and_external_seeds_have_joint_block_identity(self):
        c=self.config();c.pop('source_rng_seed');c['random_stream_policy']='independent_initial_external'
        for row,initial in zip(c['cases'],[95001,95002]):
            row.pop('source_rng_seed');row.update(initial_rng_seed=initial,external_noise_seed=96001)
        rows=case_settings(c)
        self.assertNotEqual(rows[0]['block_id'],rows[1]['block_id'])
        self.assertEqual(rows[0]['source_rng_seed'],96001)
        self.assertEqual(rows[0]['initial_rng_seed'],95001)
        c['cases'][1]['initial_rng_seed']=95001
        self.assertEqual(case_settings(c)[0]['block_id'],case_settings(c)[1]['block_id'])
        c['cases'][0]['source_rng_seed']=123
        with self.assertRaisesRegex(ValueError,'NAMED_SEEDS'):case_settings(c)
        c=self.config();c['cases'][0]['initial_rng_seed']=95001
        with self.assertRaisesRegex(ValueError,'EXPLICIT_POLICY'):case_settings(c)

    def test_matched_prefix_requires_same_initial_and_external_stream(self):
        a=self.record('a',96001,.06);b=self.record('b',96001,.06)
        a['initial_rng_seed']=95001;b['initial_rng_seed']=95002
        b['ticks'][0]=2
        self.assertEqual(verify_matched_prefixes([a,b],40),[])
        b['initial_rng_seed']=95001
        with self.assertRaisesRegex(ValueError,'PREFIX'):verify_matched_prefixes([a,b],40)

if __name__=='__main__':unittest.main()
