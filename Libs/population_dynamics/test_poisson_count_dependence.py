"""Numerical and deliberate-dependence fixtures; not native-source evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .poisson_count_dependence import (StreamingCountTrace, analyze_legacy, analyze_traces,
    covariance_diagnostic, hac_interval, independent_record_variance)


def trace(counts, means, groups=None, first_tick=0, chunk=137):
    n=counts.shape[1]
    groups=np.zeros(n,dtype=np.int64) if groups is None else np.asarray(groups,dtype=np.int64)
    stream=StreamingCountTrace(np.arange(n,dtype=np.int64),groups,np.broadcast_to(means,(n,)))
    for start in range(0,len(counts),chunk):
        value=counts[start:start+chunk]
        stream.update(value,np.arange(first_tick+start,first_tick+start+len(value)))
    return stream.finish()


class PoissonCountDependenceTests(unittest.TestCase):
    def test_streamed_populations_include_every_target_not_only_sample(self):
        rng=np.random.default_rng(101)
        x=rng.poisson(2.,(1234,40)).astype(np.int32)
        groups=np.repeat(np.arange(2),20)
        stream=StreamingCountTrace(np.arange(40),groups,np.full(40,2.),[0,21])
        for start in range(0,len(x),103):
            block=x[start:start+103]
            stream.update(block,np.arange(start+2,start+2+len(block)))
        actual=stream.finish()
        np.testing.assert_array_equal(actual.aggregate_counts[:,:2],
            np.column_stack((x[:,:20].sum(1),x[:,20:].sum(1))))
        np.testing.assert_array_equal(actual.aggregate_counts[:,-1],x.sum(1))
        np.testing.assert_array_equal(actual.sampled_counts,x[:,[0,21]])
        self.assertEqual(actual.complete_target_count,40)
        other=trace(x,np.full(40,2.),groups,first_tick=2,chunk=41)
        np.testing.assert_array_equal(actual.aggregate_counts,other.aggregate_counts)

    def test_population_reduction_accepts_actual_production_target_count(self):
        sizes=np.array([26802,7560,28399,7100,6285,1380,18654,3820])
        groups=np.repeat(np.arange(8),sizes)
        stream=StreamingCountTrace(np.arange(100000),groups,np.ones(100000))
        stream.update(np.ones((2,100000),dtype=np.int32),np.array([2,3]))
        data=stream.finish()
        np.testing.assert_array_equal(data.aggregate_counts[0,:8],sizes)
        np.testing.assert_array_equal(data.aggregate_counts[:,-1],[100000,100000])
        self.assertLessEqual(len(data.sampled_ids),64)

    def test_hac_matches_direct_bartlett_sum(self):
        x=np.random.default_rng(102).normal(size=1000)
        result=hac_interval(0.,x,7,minimum_record_spans=1)
        centered=x-x.mean()
        gamma=[np.dot(centered[:len(x)-lag],centered[lag:])/len(x) for lag in range(8)]
        expected=(gamma[0]+2*sum((1-lag/8)*gamma[lag] for lag in range(1,8)))/len(x)
        self.assertAlmostEqual(result['standard_error']**2,expected,places=14)

    def test_time_dependence_increases_uncertainty(self):
        x=np.repeat(np.random.default_rng(103).poisson(3.,1000),16).astype(float)
        naive=hac_interval(0.,x,0)
        robust=hac_interval(0.,x,64)
        self.assertGreater(robust['standard_error'],2*naive['standard_error'])

    def test_shared_count_driver_detected_by_counts_and_total_variance(self):
        x=np.random.default_rng(104).poisson(3.,12000)
        data=trace(np.column_stack((x,x)),3.)
        report=analyze_traces(data,window_ticks=(1,),temporal_lags=(1,),hac_bandwidth_ticks=32)
        pair=report['target_pairs'][0]
        self.assertGreater(pair['uncertainty']['interval'][0],.8)
        self.assertGreater(report['aggregate_variance'][-1]['variance_excess_fraction'],.8)
        self.assertFalse(report['scientific_reference_qualified'])

    def test_independent_zero_indicators_can_hide_correlated_positive_counts(self):
        rng=np.random.default_rng(105);n=30000;mean=2.
        positive=rng.poisson(mean,n)
        while np.any(positive==0):
            zero=positive==0;positive[zero]=rng.poisson(mean,int(zero.sum()))
        active=rng.random((n,2)) < -np.expm1(-mean)
        x=active*positive[:,None]
        zero_probability=np.exp(-mean)
        expected_xor=2*zero_probability*(1-zero_probability)
        observed_xor=np.mean((x[:,0]==0)!=(x[:,1]==0))
        self.assertLess(abs(observed_xor-expected_xor),.015)
        result=covariance_diagnostic(x[:,0],x[:,1],mean,mean,bandwidth=32)
        self.assertGreater(result['uncertainty']['interval'][0],.35)

    def test_negative_count_dependence_is_retained(self):
        left=np.random.default_rng(106).binomial(4,.5,12000)
        result=covariance_diagnostic(left,4-left,2.,2.,bandwidth=32)
        self.assertLess(result['uncertainty']['interval'][1],-.3)

    def test_cross_generator_lag_direction_is_explicit(self):
        z=np.random.default_rng(107).poisson(2.,6001)
        data=trace(np.column_stack((z[:-1],z[1:])),2.,[0,1])
        report=analyze_traces(data,window_ticks=(1,),temporal_lags=(1,),hac_bandwidth_ticks=16)
        lead=next(row for row in report['cross_generator_pairs']
            if row['left_group']=='generator_1' and row['right_group']=='generator_0' and row['lag_ticks']==1)
        self.assertGreater(lead['covariance_over_poisson_sd'],.85)

    def test_short_records_do_not_receive_hac_intervals(self):
        result=hac_interval(0.,np.arange(10.),3)
        self.assertIsNone(result['interval'])
        self.assertEqual(result['status'],'INSUFFICIENT_RECORD_FOR_DECLARED_HAC')

    def test_independent_record_uncertainty_does_not_count_repeated_time_as_replication(self):
        values=np.random.default_rng(108).poisson(3.,(24,2))
        short=[trace(np.tile(v,(32,1)),3.) for v in values]
        long=[trace(np.tile(v,(320,1)),3.) for v in values]
        ids=[str(i) for i in range(24)]
        left=independent_record_variance(short,ids,window_ticks=(1,))
        right=independent_record_variance(long,ids,window_ticks=(1,))
        for a,b in zip(left['aggregate_variance'],right['aggregate_variance']):
            self.assertAlmostEqual(a['variance_excess_fraction'],b['variance_excess_fraction'],places=12)
            self.assertAlmostEqual(a['standard_error'],b['standard_error'],places=12)
        with self.assertRaisesRegex(ValueError,'unique record'):
            independent_record_variance(short,['same']*24,window_ticks=(1,))

    def test_finite_support_failure_is_preserved_without_rewriting_legacy_evidence(self):
        rng=np.random.default_rng(109)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);x=rng.poisson(2.,(200,2));x[0,0]=1
            np.savez(path/'counts.npz',counts=x,arrival_ticks=np.arange(200),target_ids=np.arange(2),
                generator_index=np.array([0,1]),lambda_per_target=np.array([2.,2.]))
            config={'dt_ms':.1,'expected_arrival_support':{'first_tick':2,'last_tick':199}}
            (path/'acquisition.json').write_text(json.dumps({'status':'EXECUTED','config':config}))
            previous=json.dumps({'law_test_outcome':'REJECTED'})
            (path/'analysis.json').write_text(previous)
            result=analyze_legacy(path,window_ticks=(1,),temporal_lags=(1,),hac_bandwidth_ticks=4)
            self.assertEqual(result['status'],'SUPPORT_VIOLATION_PRESERVED')
            self.assertEqual(result['legacy_law_test_outcome'],'REJECTED')
            self.assertEqual((path/'analysis.json').read_text(),previous)

    def test_stream_rejects_overlap_and_negative_counts(self):
        stream=StreamingCountTrace([0,1],[0,0],[1.,1.])
        stream.update(np.ones((2,2),dtype=int),np.array([0,1]))
        with self.assertRaisesRegex(ValueError,'gap, overlap'):
            stream.update(np.ones((2,2),dtype=int),np.array([1,2]))
        with self.assertRaisesRegex(ValueError,'integer target'):
            stream.update(np.array([[-1,0]]),np.array([2]))


if __name__=='__main__':
    unittest.main()
