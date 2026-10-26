"""Source preference, paired-response and bound phase-readout regression tests."""
from contextlib import contextmanager
import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from . import pfc_fluctuation_comparison as comparison
from . import pfc_phase_responses as responses
from .pfc_response_analysis import summarize_counts
from .test_pfc_fluctuation_comparison import make_cohort, bind, file_row
from .provenance import sha256_file


def rebind_reference(root):
    ref=root/'reference';report=comparison.read(ref/'analysis/analysis.json')
    artifacts={name:sha256_file(ref/'analysis'/name) for name in ('responses.csv','phase_unit_counts.npz')}
    replay=comparison.read(ref/'analysis/readout_replay.json');replay['artifacts']=artifacts
    comparison.write_json(ref/'analysis/readout_replay.json',replay)
    report.update(artifacts=artifacts,readout_replay_sha256=sha256_file(ref/'analysis/readout_replay.json'))
    comparison.write_json(ref/'analysis/analysis.json',report)
    delivery=dict(status='ARCHIVED_PFC_SOURCE_READOUTS',exit_code=0,
        files=[file_row(ref,name) for name in responses.REFERENCE_FILES if name!='delivery.json'])
    comparison.write_json(ref/'delivery.json',delivery)
    upstream=comparison.read(root/'analysis/report.json');upstream['reference_delivery_sha256']=sha256_file(ref/'delivery.json')
    comparison.write_json(root/'analysis/report.json',upstream);bind(root)


def add_reference(root):
    ref=root/'reference';(ref/'analysis').mkdir(parents=True)
    config=yaml.safe_load((root/'config.yaml').read_text());upstream=comparison.read(root/'analysis/report.json')
    ne,ni=config['excitatory_neurons'],config['inhibitory_neurons'];mapping,sizes=comparison.unit_mapping(ne,ni,16)
    active=np.array([np.flatnonzero(mapping==field)[0] for field in range(17)])
    arrays,rows={},[]
    with np.load(root/'analysis/field_fluctuations.npz') as fields:
        seeds=fields['source_rng_seeds'].tolist()
        for condition in responses.CONDITIONS:
            for bi,seed in enumerate(seeds):
                case=f'pfc_{seed}_{condition}';spec=next(x for x in config['cases'] if x['case_id']==case)
                for phase,(start,stop) in responses.PHASES.items():
                    counts=np.zeros(ne+ni,dtype=np.int64)
                    counts[active]=fields[condition+'_field_counts'][bi,round(start/5):round(stop/5)].sum(axis=0)
                    arrays[case+'_'+phase]=counts
                    value=summarize_counts(counts,ne,config['stimulus_center_deg'],config['stimulus_width_deg'],stop-start)
                    value['all_unit_rate_hz']=float(counts.sum())/((ne+ni)*(stop-start)/1000)
                    rows.append(dict(case_id=case,source_rng_seed=seed,condition=condition,phase=phase,
                        start_ms=start,stop_ms=stop,units=ne+ni,excitatory_neurons=ne,inhibitory_neurons=ni,
                        stimulus_strength_nA=spec['stimulus_strength_nA'],**value))
    np.savez_compressed(ref/'analysis/phase_unit_counts.npz',**arrays)
    comparison.write_rows(ref/'analysis/responses.csv',rows)
    report=dict(status='DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC',complete_source_episodes=len(config['cases']),
        registered_source_cases=len(config['cases']),missing_source_cases=[],unresolved_input_paths=[],
        config_sha256=upstream['config_sha256'],source_bindings=upstream['source_bindings'],
        common_execution_binding=upstream['common_execution_binding'],configuration=config,source_rng_seeds=seeds,
        units=ne+ni,phases_ms=responses.PHASES,paired_effects_available=True,
        source_prefix_and_rng_pairs_verified=3*len(seeds),paired_contrasts=15*len(seeds))
    comparison.write_json(ref/'analysis/analysis.json',report)
    comparison.write_json(ref/'analysis/readout_replay.json',dict(status='PASS',config_sha256=upstream['config_sha256'],source_bindings=upstream['source_bindings']))
    rebind_reference(root)


class PhaseResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)/'clock';make_cohort(cls.root,.05);add_reference(cls.root)

    @contextmanager
    def changed(self,name,change,*,rebind=True):
        names=set(['reference/'+name,'analysis/report.json','delivery.json']+
                  ['reference/'+x for x in responses.REFERENCE_FILES if not x.endswith('.npz')])
        saved={x:(self.root/x).read_bytes() for x in names}
        try:
            change(self.root/'reference'/name)
            if rebind:rebind_reference(self.root)
            yield
        finally:
            for name,value in saved.items():(self.root/name).write_bytes(value)

    def test_true_unit_preferences_and_undefined_directions_retained(self):
        cohort=comparison.load_cohort(self.root);value=cohort['phase_responses']
        self.assertEqual(value['values'].shape,(3,3,5,8))
        self.assertFalse(value['direction_defined'][:,:,0].any())
        self.assertTrue(np.isnan(value['decoded_angle_deg'][:,:,0]).all())
        self.assertEqual(value['metadata']['complete_unit_phase_counts'],4500000)
        with np.load(self.root/'reference/analysis/phase_unit_counts.npz') as stored:
            counts=stored['pfc_94002_cue_cue'][:80000]
        indices=np.flatnonzero(counts)
        exact=math.fsum(float(counts[i])*math.cos(2*math.pi*i/80000) for i in indices)/(80000*.2)
        approximate=math.fsum(float(counts[i])*math.cos(2*math.pi*((i*16//80000)+.5)/16) for i in indices)/(80000*.2)
        got=value['values'][1,1,1,responses.METRICS.index('vector_x_hz')]
        self.assertAlmostEqual(got,exact,places=12)
        self.assertGreater(abs(got-approximate),1e-8)

    def test_changed_delivered_response_is_rejected(self):
        def change(path):path.write_bytes(path.read_bytes()+b'\n')
        with self.changed('analysis/responses.csv',change,rebind=False):
            with self.assertRaisesRegex(ValueError,'size or hash mismatch'):comparison.load_cohort(self.root)

    def test_duplicate_phase_and_invalid_pairing_evidence_are_rejected(self):
        def duplicate(path):
            with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
            rows[2]=rows[1]
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.changed('analysis/responses.csv',duplicate):
            with self.assertRaisesRegex(ValueError,'case/phase response'):comparison.load_cohort(self.root)
        def no_pairs(path):
            report=comparison.read(path);report['source_prefix_and_rng_pairs_verified']=0;comparison.write_json(path,report)
        with self.changed('analysis/analysis.json',no_pairs):
            with self.assertRaisesRegex(ValueError,'input-pair evidence'):comparison.load_cohort(self.root)

    def test_rebound_incorrect_circular_component_is_recomputed_from_unit_counts(self):
        def change(path):
            with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
            rows[1]['vector_x_spikes']=float(rows[1]['vector_x_spikes'])+10
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.changed('analysis/responses.csv',change):
            with self.assertRaisesRegex(ValueError,'readout mismatch: vector_x_spikes'):comparison.load_cohort(self.root)

    def test_signed_clock_difference_of_paired_control_responses(self):
        cohorts={};values_by_clock=([2,2,5,7,9,11],[1,2,4,3,7,8],[0,0,0,0,0,0])
        for label,clock,flat in zip(('coarse','middle','fine'),comparison.CLOCKS,values_by_clock):
            values=np.zeros((3,2,5,8));values[:,:,1,6]=np.asarray(flat).reshape(3,2)
            cohorts[label]=dict(config=dict(dt_ms=clock),report=dict(joint_rng_blocks=[1,2]),phase_responses=dict(
                values=values,direction_defined=np.zeros((3,2,5),dtype=bool),decoded_angle_deg=np.full((3,2,5),np.nan),
                resultant_length=np.zeros((3,2,5)),metadata=dict(scope='algebra_fixture')))
        tables,payload,report=responses.compare(cohorts)
        rows=[r for r in tables['paired_response_clock_differences.csv'] if r['left_cohort']=='coarse'
              and r['phase']=='cue' and r['metric']=='vector_x_hz']
        result={r['contrast']:r['left_minus_right'] for r in rows}
        self.assertEqual(result,dict(cue_minus_omission=2.,strong_cue_minus_omission=2.,strong_cue_minus_cue=0.))
        self.assertFalse(payload['coarse_direction_defined'].any())
        self.assertTrue(report['direction_undefined_is_not_exclusion'])


if __name__=='__main__':unittest.main()
