"""Synthetic native/archive contracts for the 5 s PFC delay diagnostic."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import yaml

from . import pfc_delay_analysis as analysis
from .circuit_data import CircuitArchiveWriter, CircuitEpisode, CircuitUnits, SCHEMA
from .pfc_reference_replay import compare
from .provenance import sha256_file


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def events(seed, condition):
    result=[(10,340+seed%3),(50,1030)]
    for phase,begin in (('cue',2100),('early_delay',6100),('middle_delay',20100),('late_delay',80100)):
        number=(12 if phase=='cue' else 8) if condition=='strong_cue' else 1
        if condition=='cue':
            number=3 if phase=='cue' or (phase=='early_delay' and seed!=91003) else 0
        for i in range(number):
            result.append((begin+i*40,340+i%3))
    return sorted(result)


def make_source(root, config, conditions):
    root.mkdir(parents=True)
    config=copy.deepcopy(config)
    config['cases']=[r for r in config['cases'] if r['case_id'] in {
        f'pfc_{seed}_{condition}' for seed in analysis.SEEDS for condition in conditions}]
    config_path=root/'config.yaml';config_path.write_text(yaml.safe_dump(config))
    identity=analysis.provenance(config,analysis.SOURCE_ENTRY_SHA256)
    units=CircuitUnits(analysis.CIRCUIT,np.arange(1280),np.array(['E']*1024+['I']*256),
        np.r_[np.arange(1024)*16//1024,np.full(256,16)],
        tuple([f'E_preference_{i*360/16:g}_{(i+1)*360/16:g}_deg' for i in range(16)]+['I_population']),
        'E angle=360*source_index/N_E degrees; I has no preference coordinate',
        analysis.SOURCE_REVISION,circuit_provenance=identity)
    binding=dict(schema=SCHEMA,scope='source_reproduction',scenario_id='pfc_working_memory',
        source_revision=analysis.SOURCE_REVISION,sample_unit='complete_mechanistic_episode',
        config_sha256=sha256_file(config_path),source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,
        resolved_source_config=config)
    cases=[];duration=round(config['duration_ms']/.05)
    features=({'name':'cue_current','unit':'nA','population':'E','center_degree':120.,'width_degree':30.},
        {'name':'distractor_current','unit':'nA','population':'E','center_degree':90.,'width_degree':40.})
    with CircuitArchiveWriter(root/'dataset',binding) as writer:
        writer.register_circuit(units)
        for seed in analysis.SEEDS:
            for condition in conditions:
                case_id=f'pfc_{seed}_{condition}'
                pairs=[pair for pair in events(seed,condition) if pair[0]<duration]
                ticks=np.array([p[0] for p in pairs],dtype=np.int64)
                ids=np.array([p[1] for p in pairs],dtype=np.int64)
                raw=root/(case_id+'_native.npz')
                np.savez_compressed(raw,spike_ticks=ticks,spike_unit_indices=ids,spike_seconds=ticks*.05/1000,
                    native_spike_counts=np.bincount(ids,minlength=1280),
                    terminal_voltage_mV=np.full(1280,-60.))
                receipt=dict(complete_unit_recording=True,simulation_completed=True,
                    recorded_until_tick=duration,expected_duration_ticks=duration,core_unit_count=1280,
                    source_revision=analysis.SOURCE_REVISION,circuit_provenance=identity,
                    raw_sha256=sha256_file(raw),native_source_sha256=analysis.SOURCE_ENTRY_SHA256,
                    initial_and_background_streams='source_global_rng_not_separated')
                controls=np.zeros((5,2));controls[1,0]=analysis.CONDITIONS[condition]
                writer.add(CircuitEpisode(case_id,analysis.CIRCUIT,f'pfc_source_block_{seed}',
                    'development',duration,.05,ticks,ids,np.array([0,2000,6000,8000,9000]),controls,
                    features,seeds={'source_rng_seed':seed},source_record=receipt))
                cases.append(dict(case_id=case_id,source_rng_seed=seed,
                    stimulus_strength_nA=analysis.CONDITIONS[condition],distractor_strength_nA=0.,
                    core_units=1280,native_events=len(ticks),raw_file=raw.name,raw_sha256=sha256_file(raw)))
    report=dict(status='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',scope='source_reproduction',
        source_revision=analysis.SOURCE_REVISION,source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,
        brian2_version='synthetic-fixture',numpy_version=np.__version__,
        independent_source_rng_blocks=3,full_scale_qualified=False,scientific_fidelity_qualified=False,
        resolved_config=config,cases=cases)
    analysis.write_json(root/'report.json',report)
    return report


def make_reference(root,config,conditions,status):
    config=copy.deepcopy(config);config['duration_ms']=500.
    report=make_source(root/'original',config,conditions)
    catalog=analysis.read(root/'original/dataset/dataset.json')
    names=['original/report.json','original/dataset/dataset.json']
    names.extend('original/dataset/'+r['path'] for r in catalog['circuits'].values())
    names.extend('original/'+r['raw_file'] for r in report['cases'])
    analysis.write_json(root/'delivery.json',dict(status=status,units=1280,duration_ms=500,
        source_rng_blocks=3,working_memory_qualified=False,neuroscientific_claim_qualified=False,
        production_episodes=0,files=[dict(path=n,sha256=sha256_file(root/n)) for n in names]))


class DelayAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pfc_delay_')
        self.root=Path(self.tmp.name)
        self.config=yaml.safe_load((Path(__file__).parents[1]/'configs/neural_population_pfc_delay.yaml').read_text())
        self.config_path=self.root/'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.original=self.root/'original';self.replay=self.root/'replay'
        make_source(self.original,self.config,analysis.CONDITIONS)
        shutil.copytree(self.original,self.replay)
        self.receipt=self.root/'replay.json'
        compare(self.original,self.replay,self.receipt)
        self.weak=self.root/'weak';self.strong=self.root/'strong'
        make_reference(self.weak,self.config,('cue','omission'),'ARCHIVED_REPLAYED_SOURCE_DIAGNOSTIC')
        make_reference(self.strong,self.config,('strong_cue',),'ARCHIVED_REPLAYED_CUE_STRENGTH_DIAGNOSTIC')
        self.design=dict(status='REGISTERED_BEFORE_EXECUTION',source_rng_blocks=3,
            source_rng_seeds=list(analysis.SEEDS),source_episodes=9,duration_ms=5000.,units=1280,
            cue_strengths_nA=[0.,.06,.5],cue_window_ms=[100.,300.],prefix_reference_ms=500.,
            phases_ms=analysis.PHASES,config_sha256=sha256_file(self.config_path),
            source_fidelity_qualified=False,scientific_claim_qualified=False,production_episodes=0)
        self.design_path=self.root/'design.json';analysis.write_json(self.design_path,self.design)

    def tearDown(self):
        self.tmp.cleanup()

    def run_analysis(self,name='analysis'):
        return analysis.analyze(self.original,self.replay,self.receipt,self.design_path,self.config_path,
                                self.weak,self.strong,self.root/name)

    def test_complete_native_prefix_phase_and_independent_readout_replay(self):
        result=self.run_analysis()
        self.assertEqual(result['status'],'DESCRIBED_REPLAYED_DELAY_DIAGNOSTIC')
        self.assertEqual(result['full_unit_500ms_prefixes_verified'],9)
        self.assertEqual(result['phase_unit_count_values_verified'],9*5*1280)
        self.assertEqual(result['source_rng_blocks'],3)
        self.assertEqual(result['production_episodes'],0)
        self.assertFalse(result['working_memory_qualified'])
        replay=analysis.read(self.root/'analysis/readout_replay.json')
        self.assertEqual(replay['analysis_artifacts'],result['artifacts'])
        self.assertEqual(result['readout_replay_sha256'],sha256_file(self.root/'analysis/readout_replay.json'))
        rows=table(self.root/'analysis/responses.csv')
        self.assertEqual(len(rows),45)
        self.assertEqual({r['condition'] for r in rows},set(analysis.CONDITIONS))
        contrasts=table(self.root/'analysis/paired_effects.csv')
        weak=[float(r['cue_selectivity_hz']) for r in contrasts
              if r['contrast']=='cue_minus_omission' and r['phase']=='early_delay']
        self.assertGreater(weak[0],0);self.assertGreater(weak[1],0);self.assertLess(weak[2],0)
        self.assertFalse({'ci_lower','ci_upper','p_value','q_value'}&set(rows[0]))
        for name,digest in result['artifacts'].items():
            self.assertEqual(sha256_file(self.root/'analysis'/name),digest)

    def test_500ms_prefix_changes_fail_and_later_events_do_not_redefine_prefix(self):
        a={'spike_ticks':np.array([1,9999,10000,80000]),'spike_unit_indices':np.array([0,1279,1,2])}
        b={'spike_ticks':np.array([1,9999]),'spike_unit_indices':np.array([0,1279])}
        self.assertEqual(analysis.prefix_check(a,b),2)
        a['spike_unit_indices'][1]=1278
        with self.assertRaisesRegex(ValueError,'prefix changed'):analysis.prefix_check(a,b)

    def test_unknown_condition_and_missing_case_fail(self):
        report=analysis.read(self.original/'report.json')
        for kind in ('unknown','missing'):
            changed=copy.deepcopy(report)
            if kind=='unknown':changed['cases'][0]['case_id']='pfc_91001_unregistered'
            else:changed['cases'].pop()
            with self.assertRaisesRegex(ValueError,'case set'):
                analysis.registered_cases(changed,analysis.CONDITIONS)

    def test_unregistered_phase_clock_or_cue_timing_fail(self):
        for key,value in [('dt_ms',.1),('stimulus_start_ms',90.),('excitatory_neurons',512)]:
            config=copy.deepcopy(self.config);config[key]=value
            self.config_path.write_text(yaml.safe_dump(config))
            design=copy.deepcopy(self.design);design['config_sha256']=sha256_file(self.config_path)
            with self.assertRaisesRegex(ValueError,'parameters, clock or timing'):
                analysis.check_design(design,config,self.config_path)
        design=copy.deepcopy(self.design);design['phases_ms']=dict(analysis.PHASES,late_delay=[3500,5000])
        with self.assertRaisesRegex(ValueError,'frozen'):
            analysis.check_design(design,self.config,self.config_path)

    def test_split_leakage_and_missing_archive_episode_fail(self):
        path=self.original/'dataset/dataset.json'
        catalog=analysis.read(path)
        for kind in ('split','missing'):
            changed=copy.deepcopy(catalog)
            if kind=='split':changed['shards'][0]['episodes'][0]['split']='test'
            else:changed['shards'][0]['episodes'].pop()
            analysis.write_json(path,changed)
            with self.assertRaisesRegex(ValueError,'development-only'):self.run_analysis(kind)

    def test_archive_event_clock_or_exact_control_change_points_must_match(self):
        catalog_path=self.original/'dataset/dataset.json'
        catalog=analysis.read(catalog_path);shard=catalog['shards'][0]
        shard_path=self.original/'dataset'/shard['path'];original_bytes=shard_path.read_bytes()
        for key in ('spike_ticks','control_ticks'):
            shard_path.write_bytes(original_bytes)
            with h5py.File(shard_path,'r+') as file:
                file['pfc_91001_cue'][key][1]+=1
            shard['sha256']=sha256_file(shard_path);analysis.write_json(catalog_path,catalog)
            with self.assertRaisesRegex(ValueError,'exact control change points'):
                self.run_analysis(key)

    def test_missing_native_replay_blocks_all_readouts(self):
        self.receipt.unlink()
        with self.assertRaises(FileNotFoundError):self.run_analysis()
        self.assertEqual(analysis.read(self.root/'analysis/analysis.json')['status'],'BLOCKED_INCOMPLETE_INPUTS')
        self.assertFalse((self.root/'analysis/responses.csv').exists())

    def test_failed_native_replay_is_not_accepted(self):
        value=analysis.read(self.receipt);value['status']='FAILED_REFRACTORY_SUPPORT'
        analysis.write_json(self.receipt,value)
        with self.assertRaisesRegex(ValueError,'all nine'):self.run_analysis()

    def test_changed_archived_prefix_is_rejected_even_if_delivery_is_rebound(self):
        report_path=self.weak/'original/report.json';report=analysis.read(report_path)
        case=report['cases'][0];path=self.weak/'original'/case['raw_file']
        with np.load(path,allow_pickle=False) as stored:payload={k:stored[k] for k in stored.files}
        payload['spike_ticks'][0]+=1;payload['spike_seconds']=payload['spike_ticks']*.05/1000
        np.savez_compressed(path,**payload)
        case['raw_sha256']=sha256_file(path);analysis.write_json(report_path,report)
        delivery=analysis.read(self.weak/'delivery.json')
        for row in delivery['files']:row['sha256']=sha256_file(self.weak/row['path'])
        analysis.write_json(self.weak/'delivery.json',delivery)
        with self.assertRaisesRegex(ValueError,'prefix changed'):self.run_analysis()

    def test_independent_replay_rejects_changed_paired_effect(self):
        self.run_analysis()
        path=self.root/'analysis/paired_effects.csv';rows=table(path)
        rows[0]['cue_selectivity_hz']='123.'
        path.unlink();analysis.write_rows(path,rows)
        report_path=self.root/'analysis/analysis.json';report=analysis.read(report_path)
        report['artifacts'][path.name]=sha256_file(path);analysis.write_json(report_path,report)
        with self.assertRaisesRegex(ValueError,'raw-event readout mismatch'):
            analysis.replay_readouts(self.original,self.root/'analysis',self.root/'bad_replay.json')

    def test_independent_replay_rejects_changed_phase_counts(self):
        self.run_analysis()
        path=self.root/'analysis/phase_unit_counts.npz'
        with np.load(path,allow_pickle=False) as data:values={k:data[k] for k in data.files}
        next(iter(values.values()))[0]+=1
        np.savez_compressed(path,**values)
        record=analysis.read(self.root/'analysis/analysis.json')
        record['artifacts'][path.name]=sha256_file(path)
        analysis.write_json(self.root/'analysis/analysis.json',record)
        with self.assertRaisesRegex(ValueError,'phase counts differ'):
            analysis.replay_readouts(self.original,self.root/'analysis',self.root/'bad_replay.json')

    def test_scalar_counter_reference_matches_vector_readout(self):
        for units in ([],[0,0,1,341,341,1024],[0,512]):
            counts=np.bincount(units,minlength=1280)
            expected=analysis.independent_summary(units,100.)
            actual=analysis.summarize_counts(counts,1024,120.,30.,100.)
            actual['all_unit_rate_hz']=len(units)/1280/.1
            self.assertEqual(analysis._check_readout(actual,expected),len(expected))

    def test_cli_help_and_missing_inputs_exit_status(self):
        command=[sys.executable,'-m','Libs.population_dynamics.pfc_delay_analysis']
        help_run=subprocess.run([*command,'--help'],capture_output=True,text=True)
        self.assertEqual(help_run.returncode,0);self.assertIn('--weak-reference',help_run.stdout)
        args=[]
        for name,value in [('original',self.original),('replay',self.replay),('replay-receipt',self.root/'absent.json'),
            ('design',self.design_path),('config',self.config_path),('weak-reference',self.weak),
            ('strong-reference',self.strong),('output',self.root/'cli_failed')]:
            args.extend(['--'+name,str(value)])
        result=subprocess.run([*command,*args],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(analysis.read(self.root/'cli_failed/analysis.json')['status'],'BLOCKED_INCOMPLETE_INPUTS')


if __name__=='__main__':
    unittest.main()

