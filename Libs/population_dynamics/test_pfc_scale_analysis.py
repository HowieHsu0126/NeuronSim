"""Synthetic complete archives test reference-size source/readout contracts."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from . import pfc_scale_analysis as analysis
from .circuit_data import CircuitArchiveWriter, CircuitEpisode, CircuitUnits, SCHEMA
from .provenance import canonical_json, sha256_file


def table(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def make_fixture(root):
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/neural_population_pfc_reference_size.yaml').read_text())
    fast=dict(config,synapse_execution='factorized_uniform_projections')
    native_config,fast_config=root/'native.yaml',root/'factorized.yaml'
    native_config.write_text(yaml.safe_dump(config));fast_config.write_text(yaml.safe_dump(fast))
    design=dict(status='REGISTERED_BEFORE_SOURCE_EXECUTION',scope='bounded_source_scaling_diagnostic',
        source_rng_blocks=list(analysis.SOURCE_SEEDS),training_seeds=[41,42,43],cases=9,duration_ms=5000.,units=2560,
        phases_ms=analysis.PHASES,relative_tolerance=0.,terminal_voltage_mV_max_absolute_error=1e-10,
        required_event_and_integer_arrays='exact',production_episodes=0,full_scale_qualified=False,scientific_fidelity_qualified=False)
    analysis.write_json(root/'design.json',design)
    native=root/'native';native.mkdir()
    intrinsic={k:config[k] for k in ('excitatory_neurons','inhibitory_neurons','weight_scaling_factor','dt_ms')}
    topology=dict(excitatory_kernel='source_circulant_gaussian',inhibitory_self_connections=False,
        ei_and_ie='source_all_to_all',neurons=[2048,512],source_entry_sha256=analysis.SOURCE_ENTRY_SHA256)
    digest=lambda value:hashlib.sha256(canonical_json(value).encode()).hexdigest()
    identity=dict(parameter_identity=digest(dict(intrinsic,source_entry_sha256=analysis.SOURCE_ENTRY_SHA256)),
                  connectivity_identity=digest(topology))
    units=CircuitUnits(analysis.CIRCUIT,np.arange(2560),np.array(['E']*2048+['I']*512),
        np.r_[np.arange(2048)*16//2048,np.full(512,16)],
        tuple([f'E_preference_{i*360/16:g}_{(i+1)*360/16:g}_deg' for i in range(16)]+['I_population']),
        'E angle=360*source_index/N_E degrees; I has no preference coordinate',
        analysis.SOURCE_REVISION,circuit_provenance=identity)
    binding=dict(schema=SCHEMA,scope='source_reproduction',scenario_id='pfc_working_memory',
        source_revision=analysis.SOURCE_REVISION,sample_unit='complete_mechanistic_episode',
        config_sha256=sha256_file(native_config),source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,
        resolved_source_config=config,synapse_execution='native_source')
    features=({'name':'cue_current','unit':'nA','population':'E','center_degree':120.,'width_degree':30.},
        {'name':'distractor_current','unit':'nA','population':'E','center_degree':90.,'width_degree':40.})
    cases=[];coverage={}
    with CircuitArchiveWriter(native/'dataset',binding) as writer:
        writer.register_circuit(units)
        for seed in analysis.SOURCE_SEEDS:
            for condition in analysis.CONDITIONS:
                case_id=f'pfc_{seed}_{condition}'
                events=[(10,680),(50,2050)]
                number=8 if condition=='strong_cue' else 1
                if condition=='cue':number=3 if seed!=93003 else 0
                for start in (2100,6100,20100,80100):
                    events.extend((start+i*40,681+i%3) for i in range(number))
                ticks=np.array([x[0] for x in events],dtype=np.int64)
                ids=np.array([x[1] for x in events],dtype=np.int64)
                payload=dict(spike_ticks=ticks,spike_unit_indices=ids,spike_seconds=ticks*.05/1000,
                    native_spike_counts=np.bincount(ids,minlength=2560),terminal_voltage_mV=np.full(2560,-60.))
                raw=native/(case_id+'_native.npz');np.savez_compressed(raw,**payload)
                coverage[case_id]=sum(v.size for v in payload.values())
                source_record=dict(complete_unit_recording=True,simulation_completed=True,
                    recorded_until_tick=100000,expected_duration_ticks=100000,core_unit_count=2560,
                    source_revision=analysis.SOURCE_REVISION,circuit_provenance=identity,
                    raw_sha256=sha256_file(raw),native_source_sha256=analysis.SOURCE_ENTRY_SHA256,
                    initial_and_background_streams='source_global_rng_not_separated')
                controls=np.zeros((5,2));controls[1,0]=analysis.CONDITIONS[condition]
                writer.add(CircuitEpisode(case_id,analysis.CIRCUIT,f'pfc_source_block_{seed}','development',
                    100000,.05,ticks,ids,np.array([0,2000,6000,8000,9000]),controls,features,
                    seeds={'source_rng_seed':seed},source_record=source_record))
                cases.append(dict(case_id=case_id,source_rng_seed=seed,stimulus_strength_nA=analysis.CONDITIONS[condition],
                    distractor_strength_nA=0.,core_units=2560,native_events=len(ticks),raw_file=raw.name,
                    raw_sha256=sha256_file(raw),elapsed_seconds=1.))
    report=dict(status='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',source_revision=analysis.SOURCE_REVISION,
        source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,scope='source_reproduction',
        synapse_execution='native_source',independent_source_rng_blocks=3,full_scale_qualified=False,
        scientific_fidelity_qualified=False,brian2_version='synthetic-fixture',numpy_version=np.__version__,
        resolved_config=config,cases=cases,execution_implementation_sha256=analysis.SOURCE_ENTRY_SHA256)
    analysis.write_json(native/'report.json',report)
    for name in ('factorized','factorized_replay'):
        target=root/name;shutil.copytree(native,target)
        changed=copy.deepcopy(report);changed.update(resolved_config=fast,synapse_execution='factorized_uniform_projections',
            execution_implementation_sha256='synthetic-factorized-source')
        analysis.write_json(target/'report.json',changed)
    comparison=dict(status='PASS_SOURCE_UNIFORM_PROJECTION_EQUIVALENCE',
        candidate_report_sha256=sha256_file(root/'factorized/report.json'),
        reference_bindings=[dict(path=str(native.resolve()),report_sha256=sha256_file(native/'report.json'))],
        design_sha256=sha256_file(root/'design.json'),source_revision=analysis.SOURCE_REVISION,
        source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,execution_implementation_sha256='synthetic-factorized-source',
        comparison_implementation_sha256=sha256_file(Path(analysis.pfc_source_comparison.__file__)),
        full_scale_qualified=False,scientific_fidelity_qualified=False,array_values=sum(coverage.values()),
        max_voltage_mV_difference=0.,cases=[dict(case_id=r['case_id'],source_rng_seed=r['source_rng_seed'],
            stimulus_strength_nA=r['stimulus_strength_nA'],spike_arrays_exact=True,max_voltage_mV_difference=0.,
            values=coverage[r['case_id']],native_seconds=1.,candidate_seconds=1.) for r in cases])
    analysis.write_json(root/'source_comparison.json',comparison)
    replay=dict(status='PASS',original_report_sha256=sha256_file(root/'factorized/report.json'),
        replayed_report_sha256=sha256_file(root/'factorized_replay/report.json'),source_revision=analysis.SOURCE_REVISION,
        complete_episodes=9,core_units=2560,refractory_violations=0,
        comparison_implementation_sha256=sha256_file(Path(analysis.pfc_reference_replay.__file__)),
        replay_runs_count_as_additional_samples=False,full_scale_qualified=False,scientific_fidelity_qualified=False,
        fresh_process_array_values_checked=sum(coverage.values()))
    analysis.write_json(root/'factorized_replay.json',replay)


class ReferenceSizeTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='pfc_scale_fixture_')
        self.root=Path(self.temporary.name);make_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def run_analysis(self,name='analysis',reference=None):
        r=self.root
        return analysis.analyze(r/'native',r/'factorized',r/'factorized_replay',r/'source_comparison.json',
            r/'factorized_replay.json',r/'design.json',r/'native.yaml',r/'factorized.yaml',r/name,
            reference_analysis=reference)

    def test_complete_all_unit_replay_and_stimulated_fraction(self):
        result=self.run_analysis()
        self.assertEqual(result['status'],'DESCRIBED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC')
        self.assertEqual(result['phase_unit_count_values_verified'],115200)
        self.assertEqual(result['scalar_readout_values_verified'],1035)
        self.assertEqual(result['source_rng_seeds'],[93001,93002,93003])
        self.assertFalse(result['full_scale_qualified']);self.assertEqual(result['production_episodes'],0)
        rows=table(self.root/'analysis/responses.csv');self.assertEqual(len(rows),45)
        for row in rows:
            self.assertEqual(row['cue_units'],'171')
            self.assertEqual(float(row['cue_fraction_e']),171/2048)
            expected=0 if row['condition']=='omission' else 171
            self.assertEqual(int(row['stimulated_e_units']),expected)
            self.assertEqual(float(row['stimulated_e_fraction']),expected/2048)
        effects=table(self.root/'analysis/paired_effects.csv')
        signed=[float(r['cue_selectivity_hz']) for r in effects
                if r['phase']=='early_delay' and r['contrast']=='cue_minus_omission']
        self.assertGreater(signed[0],0);self.assertGreater(signed[1],0);self.assertLess(signed[2],0)
        self.assertFalse({'p_value','q_value','ci_lower','ci_upper'}&set(effects[0]))
        self.assertEqual(result['computed_source_array_values'],analysis.read(self.root/'source_comparison.json')['array_values'])
        for name,digest in result['artifacts'].items():self.assertEqual(sha256_file(self.root/'analysis'/name),digest)

    def test_missing_or_failed_prerequisite_never_opens_arrays(self):
        path=self.root/'source_comparison.json';record=analysis.read(path)
        for kind in ('missing','failed'):
            if kind=='missing':path.unlink()
            else:analysis.write_json(path,dict(record,status='FAILED'))
            with patch.object(analysis.np,'load',side_effect=AssertionError('incomplete arrays were opened')):
                with self.assertRaisesRegex(ValueError,'both comparison receipts must complete'):
                    self.run_analysis(kind)
            self.assertEqual(analysis.read(self.root/kind/'analysis.json')['status'],'BLOCKED_INCOMPLETE_INPUTS')

    def test_changed_comparison_identity_or_tolerance_is_rejected(self):
        path=self.root/'source_comparison.json';original=analysis.read(path)
        for kind in ('report','case','difference','missing'):
            changed=copy.deepcopy(original)
            if kind=='report':changed['candidate_report_sha256']='different'
            elif kind=='case':changed['cases'][0]['source_rng_seed']=91001
            elif kind=='difference':changed['cases'][0]['max_voltage_mV_difference']=1e-8
            else:changed['cases'].pop()
            analysis.write_json(path,changed)
            with self.assertRaises(ValueError):self.run_analysis(kind)

    def test_old_size_replay_denominator_cannot_be_reused(self):
        path=self.root/'factorized_replay.json';record=analysis.read(path);record['core_units']=1280
        analysis.write_json(path,record)
        with self.assertRaisesRegex(ValueError,'fresh factorized replay'):self.run_analysis()

    def test_computed_array_coverage_is_checked(self):
        path=self.root/'source_comparison.json';record=analysis.read(path)
        record['cases'][0]['values']+=1;analysis.write_json(path,record)
        with self.assertRaisesRegex(ValueError,'computed native array coverage'):self.run_analysis()

    def test_wrong_clock_control_or_training_seed_is_rejected(self):
        native=yaml.safe_load((self.root/'native.yaml').read_text())
        fast=yaml.safe_load((self.root/'factorized.yaml').read_text())
        design=analysis.read(self.root/'design.json')
        for key,value in (('dt_ms',.1),('stimulus_width_deg',31.),('weight_scaling_factor',2.)):
            changed=copy.deepcopy(native);changed[key]=value
            with self.assertRaisesRegex(ValueError,'physical parameters'):
                analysis.config_contract(design,changed,dict(changed,synapse_execution=fast['synapse_execution']))
        changed=copy.deepcopy(design);changed['source_rng_blocks']=[41,42,43]
        with self.assertRaisesRegex(ValueError,'source design'):analysis.config_contract(changed,native,fast)

    def test_archive_event_identity_and_split_are_checked(self):
        path=self.root/'native/dataset/dataset.json';catalog=analysis.read(path);base=copy.deepcopy(catalog)
        catalog['shards'][0]['episodes'][0]['split']='test';analysis.write_json(path,catalog)
        with self.assertRaisesRegex(ValueError,'development-only'):self.run_analysis('split')
        catalog=base;shard=catalog['shards'][0]
        source=self.root/'native/dataset'/shard['path']
        with h5py.File(source,'r+') as file:file['pfc_93001_cue']['spike_ticks'][1]+=1
        shard['sha256']=sha256_file(source);analysis.write_json(path,catalog)
        with self.assertRaisesRegex(ValueError,'archive events'):self.run_analysis('event')

    def test_scalar_reference_and_source_rounding_at_both_sizes(self):
        for n_e,n_i,expected in ((1024,256,87),(2048,512,171)):
            _,profile=analysis.geometry(n_e);self.assertEqual(profile['cue_units'],expected)
            for events in ([],[0,n_e//2],[0,1,1,round(n_e/3),n_e]):
                counts=np.bincount(events,minlength=n_e+n_i)
                produced=analysis.summarize_counts(counts,n_e,120.,30.,100.)
                produced['all_unit_rate_hz']=len(events)/(n_e+n_i)/.1
                reference=analysis.scalar_readout(events,n_e,n_i,100.)
                self.assertEqual(analysis._check_readout(produced,reference),15)
        self.assertNotEqual(87/1024,171/2048)

    def test_cross_size_rows_remain_unpaired_and_expose_cue_fraction(self):
        self.run_analysis()
        rows=table(self.root/'analysis/responses.csv')
        reference=self.root/'prior';reference.mkdir()
        old=[]
        for row in rows:
            copied=dict(row);seed=int(row['source_rng_seed'])-2000
            copied.update(source_rng_seed=seed,case_id=f'pfc_{seed}_{row["condition"]}',cue_units=87)
            old.append(copied)
        analysis.write_rows(reference/'responses.csv',old)
        artifacts={'responses.csv':sha256_file(reference/'responses.csv')}
        replay=dict(status='PASS',analysis_artifacts=artifacts,original_report_sha256='synthetic-original',
            complete_episodes=9,source_rng_blocks=3,phase_unit_count_values_checked=57600,design_sha256='synthetic-design')
        analysis.write_json(reference/'readout_replay.json',replay)
        config=yaml.safe_load((self.root/'native.yaml').read_text())
        old_config=dict(config,excitatory_neurons=1024,inhibitory_neurons=256,weight_scaling_factor=2.)
        report=dict(status='DESCRIBED_REPLAYED_DELAY_DIAGNOSTIC',units=1280,source_rng_blocks=3,
            complete_source_episodes=9,phases_ms=analysis.PHASES,full_unit_500ms_prefixes_verified=9,
            source_revision=analysis.SOURCE_REVISION,production_episodes=0,working_memory_qualified=False,
            scientific_fidelity_qualified=False,neuroscientific_claim_qualified=False,artifacts=artifacts,
            original_report_sha256='synthetic-original',
            readout_replay_sha256=sha256_file(reference/'readout_replay.json'),
            configuration=old_config,phase_unit_count_values_verified=57600,design_sha256='synthetic-design',
            confidence_intervals=False,p_values=False,q_values=False,full_scale_qualified=False)
        analysis.write_json(reference/'analysis.json',report)
        output,scope=analysis.descriptive_cross_size(reference,rows,config)
        self.assertEqual(len(output),90);self.assertFalse(scope['paired_across_sizes'])
        self.assertEqual({r['cue_units'] for r in output},{87,171})
        self.assertEqual({r['source_rng_seed'] for r in output if r['units']==1280},{91001,91002,91003})
        self.assertEqual({r['source_rng_seed'] for r in output if r['units']==2560},{93001,93002,93003})
        self.assertIn('not a pure N effect',scope['interpretation'])
        report['configuration']['stimulus_width_deg']=31.
        analysis.write_json(reference/'analysis.json',report)
        with self.assertRaisesRegex(ValueError,'1280-unit source comparator'):
            analysis.descriptive_cross_size(reference,rows,config)

    def test_cli_help_and_blocked_exit(self):
        cmd=[sys.executable,'-m',analysis.__name__]
        help_run=subprocess.run([*cmd,'--help'],capture_output=True,text=True)
        self.assertEqual(help_run.returncode,0,help_run.stderr);self.assertIn('--reference-analysis',help_run.stdout)
        (self.root/'factorized_replay.json').unlink()
        mapping={'native':'native','factorized':'factorized','factorized-replay':'factorized_replay',
            'source-comparison':'source_comparison.json','fresh-replay':'factorized_replay.json',
            'design':'design.json','native-config':'native.yaml','factorized-config':'factorized.yaml','output':'cli'}
        args=[part for key,value in mapping.items() for part in ('--'+key,str(self.root/value))]
        failed=subprocess.run([*cmd,*args],capture_output=True,text=True)
        self.assertNotEqual(failed.returncode,0)
        self.assertEqual(analysis.read(self.root/'cli/analysis.json')['status'],'BLOCKED_INCOMPLETE_INPUTS')


if __name__=='__main__':
    unittest.main()

