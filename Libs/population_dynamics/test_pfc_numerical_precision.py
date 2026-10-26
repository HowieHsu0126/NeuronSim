"""Exact resampling algebra and complete-cohort precision CLI checks."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from . import pfc_numerical_precision as precision
from . import pfc_fluctuation_comparison as comparison
from .test_pfc_fluctuation_comparison import make_cohort, add_histograms
from .test_pfc_phase_responses import add_reference


def settings(seeds):
    return dict(schema=precision.SCHEMA,scope='development_calibration',
        inference='sampling_precision_only',independent_unit='joint_initial_background_rng_block',
        joint_rng_seeds=seeds,bootstrap_replicates=16,bootstrap_seed=97001)


def cohort_registration(seeds,clock):
    return dict(config=dict(dt_ms=clock,source_replication=dict(purpose='numerical_calibration',
        independent_unit='joint_initial_background_rng_block',joint_rng_seeds=seeds)),
        report=dict(joint_rng_blocks=seeds,completed_source_cases=3*len(seeds),registered_source_cases=3*len(seeds)))


class PrecisionAlgebraTests(unittest.TestCase):
    def test_exhaustive_negative_lag_covariance_and_analytic_bootstrap_center(self):
        counts=np.array([[2,0],[1,1],[1,1],[0,2]])
        rates=np.array([[[0.],[2.]],[[2.],[0.]]])
        draws=precision.resampled_lag_covariance(rates,1,counts)[:,0,0]
        np.testing.assert_allclose(draws,[0,-2,-2,0],atol=1e-12)
        original=comparison.lag_covariance(rates,1)[0,0]
        self.assertEqual(original,-2.)
        summary=precision.PrecisionSummary(4)
        row=summary.add([original],draws[:,None],[original/2],[dict(endpoint='lag')])[0]
        self.assertEqual(row['bootstrap_mean'],-1.)
        self.assertEqual(row['analytic_bootstrap_bias'],1.)
        self.assertAlmostEqual(row['bootstrap_standard_error'],np.sqrt(4/3))
        maximum,report=summary.finish()
        np.testing.assert_allclose(maximum,np.full(4,np.sqrt(3/4)))
        self.assertFalse(report['maximum_is_simultaneous_confidence_statement'])

    def test_shared_resamples_linear_cdf_and_joint_maximum(self):
        counts=np.array([[2,0],[1,1],[1,1],[0,2]])
        covariance=np.array([0.,-2.,-2.,0.])
        cdf=counts/2 @ np.array([-.5,.5])
        np.testing.assert_array_equal(cdf,[-.5,0,0,.5])
        summary=precision.PrecisionSummary(4)
        summary.add([-2.],covariance[:,None],[-1.],[dict(endpoint='covariance')])
        row=summary.add([0.],cdf[:,None],[0.],[dict(endpoint='cdf')])[0]
        self.assertEqual(row['bootstrap_bias_estimate'],0.)
        self.assertAlmostEqual(row['bootstrap_standard_error'],np.sqrt(1/6))
        maximum,report=summary.finish()
        np.testing.assert_allclose(maximum,[np.sqrt(1.5),np.sqrt(.75),np.sqrt(.75),np.sqrt(1.5)])
        self.assertEqual(report['positive_spread_endpoints'],2)

    def test_nonidentical_clocks_preserve_signed_endpoint_pipeline(self):
        seeds=[95001,95002,95003]
        cohorts={label:cohort_registration(seeds,clock) for label,clock in
                 zip(('coarse','middle','fine'),comparison.CLOCKS)}
        cdfs=dict(source_rng_seeds=np.array(seeds),window_ms=np.array([5]),count_support=np.array([0,1,2]))
        covariances={}
        for label,cohort in cohorts.items():
            cohort['sizes']=np.array([1])
            counts=np.broadcast_to(np.arange(3)[:,None,None],(3,1000,1)).copy() if label=='middle' else np.zeros((3,1000,1),dtype=int)
            cohort['fields']={condition+'_field_counts':counts for condition in precision.CONDITIONS}
            cdf=np.array([[1,1,1],[0,1,1],[0,0,1]],dtype=float) if label=='middle' else np.ones((3,3))
            cdfs[label+'_cdf']=np.broadcast_to(cdf[None,:,None,None,None,:],(3,3,5,1,1,3)).copy()
            for condition in precision.CONDITIONS:
                for phase in precision.PHASES:
                    covariances[label,condition,phase,0]=np.array([[40000. if label=='middle' else 0.]])
        tables,payload,report=precision.analyze(cohorts,cdfs,covariances,[0],settings(seeds))
        raw=[np.arange(3)[np.repeat(np.arange(3),row)] for row in payload['bootstrap_multiplicities']]
        mean_draws=np.array([sample.mean()*200 for sample in raw])
        covariance_draws=np.array([(sample*200).var(ddof=1) for sample in raw])
        for label,sign in (('coarse',-1),('middle',1)):
            mean=next(row for row in tables['field_mean_sampling_precision.csv'] if row['left_cohort']==label)
            covariance=next(row for row in tables['field_lag_covariance_sampling_precision.csv'] if row['left_cohort']==label)
            self.assertAlmostEqual(mean['estimate'],sign*200.)
            self.assertAlmostEqual(mean['bootstrap_mean'],sign*mean_draws.mean())
            self.assertAlmostEqual(covariance['estimate'],sign*40000.)
            self.assertAlmostEqual(covariance['analytic_bootstrap_mean'],sign*40000.*2/3)
            self.assertAlmostEqual(covariance['bootstrap_mean'],sign*covariance_draws.mean())
            self.assertAlmostEqual(covariance['bootstrap_standard_error'],covariance_draws.std(ddof=1))
        cdf=next(row for row in tables['count_cdf_sampling_precision.csv'] if row['left_cohort']=='coarse' and row['count_threshold']==0)
        self.assertAlmostEqual(cdf['estimate'],2/3)
        self.assertAlmostEqual(cdf['bootstrap_mean'],np.mean([1-np.mean(sample<=0) for sample in raw]))
        self.assertEqual(report['diagnostics']['positive_spread_endpoints'],120)
        self.assertEqual(report['diagnostics']['empirically_degenerate_endpoints'],30)
        self.assertTrue(np.isfinite(payload['max_standardized_empirical_fluctuation']).all())

    def test_common_transient_and_undefined_phase_lag(self):
        x=np.repeat(np.arange(12,dtype=float).reshape(1,6,2),3,axis=0)
        counts=np.array([[3,0,0],[1,1,1],[0,1,2]])
        for lag in (0,1,5):
            np.testing.assert_array_equal(precision.resampled_lag_covariance(x,lag,counts),np.zeros((3,2,2)))
        self.assertIsNone(precision.resampled_lag_covariance(x,6,counts))

    def test_zero_empirical_spread_and_missing_lag_are_not_certainty(self):
        summary=precision.PrecisionSummary(4)
        row=summary.add([.2],np.full((4,1),.2),[.2],[dict(endpoint='constant')])[0]
        self.assertEqual(row['status'],'EMPIRICALLY_DEGENERATE')
        self.assertEqual(row['bootstrap_standard_error'],0.)
        missing=summary.add_undefined([dict(endpoint='undefined')])[0]
        self.assertIsNone(missing['estimate'])
        maximum,report=summary.finish()
        self.assertTrue(np.isnan(maximum).all())
        self.assertEqual(report['requested_endpoints'],2)
        self.assertEqual(report['maximum_excludes_endpoints'],2)
        self.assertFalse(report['zero_empirical_spread_implies_zero_population_uncertainty'])

    def test_invalid_draws_and_changed_replication_are_rejected(self):
        summary=precision.PrecisionSummary(2)
        with self.assertRaisesRegex(ValueError,'every bootstrap draw'):
            summary.add([0.],[[0.],[np.nan]],[0.],[dict(endpoint='invalid')])
        seeds=[95001,95002,95003,95004]
        cohorts={str(dt):cohort_registration(seeds,dt) for dt in comparison.CLOCKS}
        self.assertEqual(precision.validate_settings(settings(seeds),cohorts),seeds)
        for change in ('missing','confirmation','blocks'):
            bad=deepcopy(cohorts);first=next(iter(bad.values()))
            if change=='missing':first['report']['completed_source_cases']-=1
            elif change=='confirmation':first['config']['source_replication']['purpose']='confirmation'
            else:first['report']['joint_rng_blocks']=seeds[:-1]
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'complete declared'):
                precision.validate_settings(settings(seeds),bad)


class PrecisionArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temporary.cleanup)
        cls.root=Path(cls.temporary.name)
        cls.clocks=[cls.root/('clock_'+str(i)) for i in range(3)]
        cls.seeds=[95001,95002,95003,95004]
        for root,dt in zip(cls.clocks,comparison.CLOCKS):
            make_cohort(root,dt,seeds=cls.seeds);add_histograms(root);add_reference(root)
        cls.native=cls.root/'native';make_cohort(cls.native,.05,native=True)
        cls.config=cls.root/'comparison.yaml'
        cls.config.write_text(yaml.safe_dump(dict(scope='development_calibration',bin_ms=5.,
            lag_ms=[0,5,10,100],lag_scope='complete_within_phase_pairs',inference='descriptive_only',count_windows_ms=[5,50,100])))
        cls.precision=cls.root/'precision.yaml';cls.precision.write_text(yaml.safe_dump(settings(cls.seeds)))

    def command(self,output,precision_config):
        command=[sys.executable,'-B','-m',comparison.__name__,'--config',str(self.config),
            '--precision-config',str(precision_config),'--native-reference',str(self.native),'--output',str(output)]
        for root in self.clocks:command+=['--clock-archive',str(root)]
        return command

    def test_complete_four_block_cli_retains_all_endpoints_and_degeneracy(self):
        output=self.root/'complete'
        run=subprocess.run(self.command(output,self.precision),capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        report=comparison.read(output/'report.json');result=report['calibration_precision']
        self.assertEqual(report['source_episodes_reused'],45)
        self.assertEqual(result['source_episodes_reused'],36)
        self.assertEqual(result['independent_joint_blocks'],4)
        self.assertEqual(report['row_counts']['field_mean_sampling_precision.csv'],510)
        self.assertEqual(report['row_counts']['field_lag_covariance_sampling_precision.csv'],34680)
        self.assertEqual(report['row_counts']['count_cdf_sampling_precision.csv'],report['row_counts']['count_cdf_differences.csv'])
        self.assertEqual(report['row_counts']['source_response_sampling_precision.csv'],240)
        self.assertEqual(report['row_counts']['paired_response_sampling_precision.csv'],240)
        self.assertTrue(result['direction_undefined_blocks_retained'])
        self.assertEqual(result['diagnostics']['undefined_lag_endpoints'],1734)
        self.assertEqual(result['diagnostics']['positive_spread_endpoints'],0)
        for key in ('confidence_intervals','p_values','q_values','confirmation_execution_released',
                    'numerical_equivalence_qualified','scientific_fidelity_qualified'):
            self.assertFalse(result[key])
        with np.load(output/'calibration_bootstrap.npz') as stored:
            self.assertEqual(stored['bootstrap_multiplicities'].shape,(16,4))
            np.testing.assert_array_equal(stored['bootstrap_multiplicities'].sum(axis=1),np.full(16,4))
            np.testing.assert_array_equal(stored['source_rng_seeds'],self.seeds)
            self.assertTrue(np.isnan(stored['max_standardized_empirical_fluctuation']).all())

    def test_actual_32_block_protocol_rejects_partial_cohort_before_bootstrap(self):
        canonical=Path(__file__).parents[1]/'configs/neural_population_pfc_numerical_precision.yaml'
        output=self.root/'incomplete'
        run=subprocess.run(self.command(output,canonical),capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        report=comparison.read(output/'report.json')
        self.assertEqual(report['status'],'FAILED')
        self.assertIn('complete declared numerical-calibration',report['error'])
        self.assertFalse((output/'calibration_bootstrap.npz').exists())


if __name__=='__main__':
    unittest.main()
