"""Small source-archive fixtures for the paired distractor readout contract."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import yaml

from . import pfc_distractor_response as response
from . import pfc_scale_analysis as scale
from . import test_pfc_source_scale_analysis as fixtures
from .provenance import sha256_file


class DistractorResponseTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SourceReadoutTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.fixture.config.update(distractor_center_deg=90.,distractor_width_deg=40.,
            distractor_start_ms=400.,distractor_duration_ms=50.,readout_bin_ms=5.)
        self.fixture.config_path.write_text(yaml.safe_dump(self.fixture.config))
        self.baseline=self.fixture.add_current(self.fixture.archive('pfc_94001_strong_cue',suffix='_baseline'))
        self.candidate=self.fixture.add_current(self.fixture.archive('pfc_94001_strong_cue',suffix='_candidate'))
        self.install_current(self.baseline,0.)
        self.install_current(self.candidate,.5)
        self.config=copy.deepcopy(self.fixture.config)
        case=dict(next(row for row in self.config['cases'] if row['case_id']=='pfc_94001_strong_cue'),
                  case_id='pfc_94001_strong_cue_distractor',distractor_strength_nA=.5)
        self.config['cases']=[case]
        self.config['distractor_response']=dict(status='REGISTERED_BEFORE_SOURCE_EXECUTION',
            scientific_decision='descriptive_no_success_threshold',independent_joint_rng_blocks=1,
            baseline_case_id='pfc_94001_strong_cue',candidate_case_id=case['case_id'],
            baseline_delivery_sha256=sha256_file(self.baseline.parent/'delivery.json'),
            baseline_config_sha256=sha256_file(self.baseline/'config.yaml'),
            expected_coupled_units=20,distractor_strength_nA=.5)
        self.config_path=self.root/'distractor.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))
        (self.candidate/'config.yaml').write_bytes(self.config_path.read_bytes())
        report=scale.read(self.candidate/'report.json')
        report.update(case_id=case['case_id'],selected_case=case,resolved_config=self.config,
            registered_case_ids=[case['case_id']],registered_case_count=1,
            config_sha256=sha256_file(self.config_path))
        request=scale.read(self.candidate/'request.json')
        for key in ('case_id','selected_case','registered_case_ids','registered_case_count','config_sha256'):
            request[key]=report[key]
        self.fixture.write_json(self.candidate/'request.json',request)
        self.fixture.write_json(self.candidate/'report.json',report)
        delivery=scale.read(self.candidate.parent/'delivery.json');delivery['case_id']=case['case_id']
        self.fixture.write_json(self.candidate.parent/'delivery.json',delivery)
        # Move one late E event, retaining all per-unit counts and the paired prefix.
        raw=self.arrays(self.candidate/'raw.npz')
        raw['spike_ticks'][raw['spike_ticks']==20200]=20300
        order=np.lexsort((raw['spike_unit_indices'],raw['spike_ticks']))
        for key in ('spike_ticks','spike_unit_indices'):raw[key]=raw[key][order]
        raw['spike_seconds']=raw['spike_ticks']*5e-5
        np.savez(self.candidate/'raw.npz',**raw)
        self.fixture.rebind(self.candidate)

    @staticmethod
    def arrays(path):
        with np.load(path,allow_pickle=False) as stored:return {key:stored[key] for key in stored.files}

    def install_current(self,source,strength):
        values=np.zeros((5,16),dtype=np.float64);values[1,4:7]=.5e-9;values[3,3:6]=strength*1e-9
        np.savez(source/'applied_current_observation.npz',source_unit_indices=np.arange(16,dtype=np.int64),
            archive_unit_indices=np.arange(16,dtype=np.int64),
            segment_start_ticks=np.array([0,2000,6020,8000,9000],dtype=np.int64),
            segment_stop_ticks=np.array([2000,6020,8000,9000,100000],dtype=np.int64),
            cue_active=np.array([False,True,False,False,False]),
            distractor_active=np.array([False,False,False,True,False]),current_A=values)
        self.fixture.rebind(source)

    def test_real_cli_preserves_complete_pair_and_disjoint_readouts(self):
        output=self.root/'result'
        result=subprocess.run([sys.executable,'-m',response.__name__,'--config',str(self.config_path),
            '--baseline',str(self.baseline.parent),'--candidate',str(self.candidate.parent),
            '--output',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=scale.read(output/'report.json')
        self.assertEqual(report['status'],'COMPLETE_PAIRED_DISTRACTOR_RESPONSE')
        self.assertEqual(report['actual_distractor_interval_ms'],[400.,450.])
        self.assertEqual(report['source_geometry']['disjoint_E_unit_counts'],
                         dict(cue_only=1,distractor_only=1,overlap=2,neither=12))
        self.assertEqual(report['independent_joint_rng_blocks'],1)
        self.assertFalse(report['pairing']['full_external_stream_arrays_compared'])
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertEqual(report['production_episodes'],0)
        with (output/'paired_difference.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),1000)
        self.assertTrue(any(float(row['cue_selectivity_hz_difference'])!=0 for row in rows))
        self.assertEqual(rows[0]['decoded_angle_deg_circular_change'],'')
        self.assertEqual(report['post_removal_summaries']['candidate']['first_complete_bin_start_ms'],450.)

    def test_preflight_never_consumes_a_candidate_and_preserves_nine_case_baseline(self):
        report=response.run(self.config_path,self.baseline.parent,self.root/'preflight',check_only=True)
        self.assertEqual(report['status'],'READY_SOURCE_DISTRACTOR_PROTOCOL')
        self.assertEqual(len(scale.read(self.baseline/'report.json')['resolved_config']['cases']),9)
        with self.assertRaisesRegex(ValueError,'preflight cannot consume'):
            response.run(self.config_path,self.baseline.parent,self.root/'bad_preflight',
                         candidate=self.candidate.parent,check_only=True)
        with self.assertRaisesRegex(ValueError,'all nine'):
            scale._source_registration(self.config_path)

    def test_protocol_digest_and_additional_control_changes_are_rejected(self):
        for label in ('digest','control'):
            with self.subTest(label=label):
                changed=copy.deepcopy(self.config)
                if label=='digest':changed['distractor_response']['baseline_delivery_sha256']='0'*64
                else:changed['stimulus_center_deg']=121.
                path=self.root/(label+'.yaml');path.write_text(yaml.safe_dump(changed))
                with self.assertRaises(ValueError):
                    response.run(path,self.baseline.parent,self.root/(label+'_output'),check_only=True)

    def pair_inputs(self):
        registered=response.registration(self.config_path,self.baseline.parent)
        raw,common,paired,current=response.load_baseline(registered)
        _,header=scale._source_header(self.candidate.parent,registered['candidate'],
            registered['candidate_registry'],registered['candidate_sha256'])
        other,other_common,other_paired=scale._source_values(header,registered['candidate'],
            registered['candidate_registry'],registered['dimensions'],registered['candidate_sha256'])
        return raw,other,common,other_common,paired,other_paired,current,response.observed_arrays(header)

    def test_pairing_rejects_changed_prefix_initial_rng_and_absent_current_removal(self):
        original=self.pair_inputs()
        for change in ('prefix','initial','rng','removal'):
            with self.subTest(change=change):
                args=copy.deepcopy(original)
                if change=='prefix':args[1]['spike_ticks'][0]+=1
                elif change=='initial':args[5]['initial'][0]+=1
                elif change=='rng':args[5]['final_rng']['position']+=1
                else:
                    args[6]['current_A'][-1,0]=1e-9
                    args[7]['current_A'][-1,0]=1e-9
                with self.assertRaises(ValueError):response.verify_pair(*args)

    def test_zero_amplitude_flags_are_not_current_transitions_and_angles_stay_undefined(self):
        registered=response.registration(self.config_path,self.baseline.parent)
        raw,_,_,current=response.load_baseline(registered)
        current['segment_start_ticks'][3]=8020
        current['segment_stop_ticks'][2]=8020
        rows,_=response.region_rows(raw,self.config,registered['dimensions'],current,8000,9000)
        self.assertTrue(rows[60]['source_current_transition_within_bin'])
        self.assertFalse(rows[80]['source_current_transition_within_bin'])
        self.assertIsNone(rows[0]['decoded_angle_deg'])
        self.assertIsNone(rows[0]['circular_distractor_error_deg'])
        self.assertEqual(sum(row['excitatory_spikes']+row['inhibitory_spikes'] for row in rows),
                         len(raw['spike_ticks']))


if __name__=='__main__':
    unittest.main()
