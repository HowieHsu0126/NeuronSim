"""Meaningful source-domain and independent-block regressions for design inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from .pfc_calibration_design import describe, endpoint_domain, maximum_count, response_tables


class DesignInputTests(unittest.TestCase):
    def test_right_open_refractory_support_in_all_registered_clocks(self):
        for dt in (.05, .025, .0125):
            self.assertEqual(maximum_count(4., dt, 2.), 2)
            self.assertEqual(maximum_count(5., dt, 2.), 3)
            self.assertEqual(maximum_count(50., dt, 2.), 25)
            self.assertEqual(maximum_count(100., dt, 1.), 100)
        with self.assertRaisesRegex(ValueError, 'integral source ticks'):
            maximum_count(5.01, .05, 2.)

    def test_observed_zero_is_structural_only_above_population_specific_support(self):
        row = dict(status='EMPIRICALLY_DEGENERATE', phase='cue', field_index='0',
                   window_ms='5', count_threshold='2', estimate='0', bootstrap_standard_error='0')
        self.assertEqual(endpoint_domain('count_cdf',row,.05,{'E':2.,'I':1.})[0], 'EMPIRICALLY_DEGENERATE_UNCERTAIN')
        row['count_threshold']='3'
        self.assertEqual(endpoint_domain('count_cdf',row,.05,{'E':2.,'I':1.})[0], 'STRUCTURAL_ZERO_CONTRAST')
        row['field_index']='16'
        self.assertEqual(endpoint_domain('count_cdf',row,.05,{'E':2.,'I':1.})[0], 'EMPIRICALLY_DEGENERATE_UNCERTAIN')
        row['count_threshold']='5';row['estimate']='0.01'
        with self.assertRaisesRegex(ValueError, 'structural zero contradicts'):
            endpoint_domain('count_cdf',row,.05,{'E':2.,'I':1.})

    def test_zero_late_response_cannot_inherit_pre_cue_identity(self):
        row=dict(status='EMPIRICALLY_DEGENERATE',phase='baseline',estimate='0',bootstrap_standard_error='0')
        self.assertEqual(endpoint_domain('paired_response',row,.05,{'E':2.,'I':1.})[0], 'STRUCTURAL_ZERO_CONTRAST')
        row['phase']='late_delay'
        self.assertEqual(endpoint_domain('paired_response',row,.05,{'E':2.,'I':1.})[0], 'EMPIRICALLY_DEGENERATE_UNCERTAIN')

    def test_undefined_lag_never_becomes_zero(self):
        row=dict(status='UNDEFINED_PHASE_CONTAINED_LAG',phase='baseline',lag_ms='100',estimate='')
        self.assertEqual(endpoint_domain('field_lag_covariance',row,.05,{'E':2.,'I':1.})[0], 'UNDEFINED_NO_WITHIN_PHASE_PAIRS')
        row['estimate']='0'
        with self.assertRaisesRegex(ValueError,'must remain unestimated'):
            endpoint_domain('field_lag_covariance',row,.05,{'E':2.,'I':1.})

    def test_signed_responses_and_block_scatter_preserve_replication(self):
        x=np.zeros((3,3,5,8));x[2,:,4,6]=[-1.,0.,1.]
        config=dict(excitatory_neurons=2048,inhibitory_neurons=512,dt_ms=.05)
        blocks,scales,paired=response_tables('native_2560',x,[93001,93002,93003],config)
        row=next(r for r in scales if r['condition']=='strong_cue' and r['phase']=='late_delay' and r['metric']=='vector_x_hz')
        self.assertEqual((row['mean'],row['block_sd'],row['joint_rng_blocks']),(0.,1.,3))
        self.assertEqual(len(blocks),360);self.assertEqual(len(paired),120)
        self.assertEqual({r['source_rng_seed'] for r in blocks},{93001,93002,93003})
        x[2,0,0,0]=1.
        with self.assertRaisesRegex(ValueError,'pre-cue source responses differ'):
            response_tables('native_2560',x,[93001,93002,93003],config)

    def test_failure_receipt_keeps_confirmation_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=root/'config.yaml';out=root/'analysis'
            cfg.write_text(yaml.safe_dump(dict(schema='invalid',scope='development_calibration')))
            with self.assertRaisesRegex(ValueError,'development design inputs required'):
                describe(cfg,root,out)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'FAILED')
            self.assertFalse(report['confirmation_execution_released'])
            self.assertFalse(report['numerical_margins_frozen'])
            with self.assertRaises(FileExistsError):describe(cfg,root,out)


if __name__ == '__main__':
    unittest.main()
