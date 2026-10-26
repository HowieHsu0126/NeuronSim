"""Tiny engineering archives for source-readout integrity, never scientific evidence."""
from __future__ import annotations

import contextlib
import hashlib
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from . import pfc_scale_analysis as analysis
from .provenance import sha256_file


class SourceReadoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root/'registered.yaml'
        # This nonexecutable engineering archive must not broaden production
        # legacy admission. Its hash is authorized only for this test lifetime.
        fixture_executor = hashlib.sha256(
            b'# Engineering fixture, not executable source: pfc_deterministic_replay.py\n').hexdigest()
        legacy_patch = mock.patch.object(analysis, '_LEGACY_DEFAULT_KERNEL_EXECUTORS',
            analysis._LEGACY_DEFAULT_KERNEL_EXECUTORS | {fixture_executor})
        legacy_patch.start(); self.addCleanup(legacy_patch.stop)
        self.config = dict(scope='source_reproduction', expected_source_revision=analysis.SOURCE_REVISION,
            synapse_execution='factorized_uniform_projections', excitatory_neurons=16, inhibitory_neurons=4,
            weight_scaling_factor=128., dt_ms=.05, duration_ms=5000., stimulus_center_deg=120.,
            stimulus_width_deg=30., stimulus_start_ms=100., stimulus_duration_ms=200., cases=[
                dict(case_id=f'pfc_{seed}_{condition}', source_rng_seed=seed,
                     stimulus_strength_nA=strength, distractor_strength_nA=0.)
                for seed in (94001,94002,94003) for condition,strength in analysis.CONDITIONS.items()])
        self.config_path.write_text(yaml.safe_dump(self.config))
        _, self.registered, self.seeds, self.dimensions = analysis._source_registration(self.config_path)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')

    @staticmethod
    def file_row(root, name):
        path = root/name
        return dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))

    def archive(self, case_id, *, suffix='', failed=False):
        root=self.root/(case_id+suffix)
        source=root/'source_run'; source.mkdir(parents=True)
        case=next(row for row in self.config['cases'] if row['case_id']==case_id)
        if failed:
            report=dict(status='PFC_SOURCE_EXECUTION_FAILED', source_execution_complete=False,
                case_id=case_id, resolved_config=self.config, selected_case=case, scientific_fidelity_qualified=False,
                production_episodes=0, failure=dict(type='RuntimeError', message='engineering fixture failure'))
            self.write_json(source/'report.json', report)
            self.write_json(root/'delivery.json', dict(status='ARCHIVED_PFC_SOURCE_EXECUTION',
                result_status=report['status'], source_execution_complete=False, case_id=case_id,
                scientific_fidelity_qualified=False, production_episodes=0,
                files=[self.file_row(root,'source_run/report.json')]))
            return source
        seed,condition=self.registered[case_id]
        code={}
        for key,name in analysis._SOURCE_CODE_FILES.items():
            p=root/'Libs/population_dynamics'/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('# Engineering fixture, not executable source: '+name+'\n')
            code[key]=sha256_file(p)
        qroot=root/'qualification';(qroot/'comparison').mkdir(parents=True)
        qualifier=dict(status='PASS_NATIVE_STREAM_SOURCE_EXECUTION', source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,
            backend='numpy', executor_sha256=code['executor_sha256'], drive_sha256=code['drive_sha256'],
            input_replay_status='PASS_NATIVE_NUMPY_DRIVE_REPLAY', input_replay_passed_comparisons=24,
            global_cases_passed=2, global_arrays_passed=10, scientific_fidelity_qualified=False,
            full_scale_qualified=False, production_episodes=0, numpy_version='engineering_fixture', validated_dt_ms=[.05])
        self.write_json(qroot/'comparison/qualification.json',qualifier)
        self.write_json(qroot/'delivery.json',dict(status='ARCHIVED_PFC_STREAM_VALIDATION',
            result_status=qualifier['status'],files=[self.file_row(qroot,'comparison/qualification.json')]))
        qualification=dict(directory=str(qroot), receipt_file='comparison/qualification.json', qualification=qualifier,
            receipt_sha256=sha256_file(qroot/'comparison/qualification.json'), delivery_sha256=sha256_file(qroot/'delivery.json'))
        source.joinpath('config.yaml').write_bytes(self.config_path.read_bytes())
        events=[(100,0),(200,17)]
        amount={'omission':0,'cue':1,'strong_cue':2}[condition]
        for start in (2000,6000,20000,80000):
            events.append((start+100,17))
            events.extend((start+100+j*100,5) for j in range(amount))
        events.sort();ticks=np.array([t for t,_ in events],dtype=np.int64)
        ids=np.array([i for _,i in events],dtype=np.int64)
        counts=np.bincount(ids,minlength=20)
        terminal={}
        fields={'E':('v','s_AMPA','s_GABA','s_NMDA_total','s_NMDA','x','I_stim','lastspike','not_refractory'),
                'I':('v','s_AMPA','s_GABA','s_NMDA_total','lastspike','not_refractory')}
        for pop,size,offset in (('E',16,0),('I',4,16)):
            for name in fields[pop]:
                terminal[pop+'__'+name]=np.zeros(size,dtype=np.float64)
            terminal[pop+'__v'][:]=-.06
            terminal[pop+'__not_refractory']=np.ones(size,dtype=np.bool_)
            terminal[pop+'__source_indices']=np.arange(size,dtype=np.int64)
            terminal[pop+'__archive_indices']=np.arange(size,dtype=np.int64)+offset
            terminal[pop+'__spike_counts']=counts[offset:offset+size]
        np.savez(source/'raw.npz',spike_ticks=ticks,spike_unit_indices=ids,spike_seconds=ticks*5e-5,
            native_spike_counts=counts,terminal_voltage_mV=np.r_[terminal['E__v'],terminal['I__v']]/.001)
        np.savez(source/'terminal_state.npz',**terminal)
        np.savez(source/'units.npz',unit_ids=np.arange(20,dtype=np.int64),
            source_unit_indices=np.r_[np.arange(16,dtype=np.int64),np.arange(4,dtype=np.int64)],
            population_index=np.r_[np.zeros(16,dtype=np.uint8),np.ones(4,dtype=np.uint8)],
            excitatory_preference_unit_ids=np.arange(16,dtype=np.int64),
            excitatory_preferred_angle_deg=360.*np.arange(16,dtype=np.float64)/16)
        np.save(source/'initial_voltage.npy',np.full(20,-60.+seed/1e6,dtype=np.float64))
        np.savez(source/'final_rng.npz',keys=np.array([seed,2,3],dtype=np.uint32),position=np.array(4,dtype=np.int64))
        self.write_json(source/'unit_metadata.json',dict(unit_file='units.npz',unit_count=20,
            population_index_labels={'0':'E','1':'I'},inhibitory_preference_coordinate=None,
            anatomical_soma_coordinates_provided=False,initial_voltage_file='initial_voltage.npy',initial_voltage_units='mV',
            terminal_state_units=dict(v='V',I_stim='A',lastspike='s',synaptic_state_variables='dimensionless',not_refractory='boolean'),
            coupling=dict(normalization=128.,representation='implicit source rules; no independent network tiling')))
        totals={'E':100,'I':25}
        driver=dict(n_e=16,n_i=4,dt_seconds=5e-5,total_ticks=100000,ticks_generated=100000,
            seeds={'source_rng_seed':seed},random_stream_policy='source_global',source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,
            rng='numpy.random.RandomState(MT19937)',initial_draw_order='I_then_E',initial_voltage_archive_order='E_then_I',
            packed_draw_order='tick_major_I_then_E',binomial_n=1000,binomial_probability=1.4*5e-5,
            external_rate_Hz=1.4,returned_dtype='uint16',numpy_version='engineering_fixture',external_event_totals=totals)
        request=dict(status='PFC_SOURCE_EXECUTION_REQUESTED',case_id=case_id,selected_case=case,
            registered_case_ids=[c['case_id'] for c in self.config['cases']],registered_case_count=len(self.registered),
            config_sha256=sha256_file(self.config_path),backend='numpy',dimensions=self.dimensions,code=code,
            backend_qualification=qualification)
        self.write_json(source/'request.json',request)
        report=dict(request,status='PFC_SOURCE_EXECUTION_COMPLETE',source_execution_complete=True,resolved_config=self.config,
            selected_case_count=1,completed_case_count=1,source_entry_sha256=analysis.SOURCE_ENTRY_SHA256,
            scientific_fidelity_qualified=False,working_memory_qualified=False,fidelity_at_requested_scale_qualified=False,
            learned_simulator=False,production_episodes=0,validations={key:dict(passed=True) for key in analysis._SOURCE_EXECUTION_CHECKS},
            seeds={'source_rng_seed':seed},random_stream_policy='source_global',driver=driver,completed_duration_ticks=100000,
            neural_event_count=len(ids),source_external_event_totals=totals,
            inputs_used=dict(input_mode='native_numpy_stream',ticks_consumed=100000,
                             E=dict(values=1600000,events=100),I=dict(values=400000,events=25)))
        report['outputs']={name:self.file_row(source,name) for name in analysis._SOURCE_OUTPUTS}
        self.write_json(source/'report.json',report)
        names=['source_run/'+name for name in sorted(analysis._SOURCE_OUTPUTS)]+['source_run/report.json',
            'qualification/delivery.json','qualification/comparison/qualification.json']+[
            'Libs/population_dynamics/'+name for name in analysis._SOURCE_CODE_FILES.values()]
        self.write_json(root/'delivery.json',dict(status='ARCHIVED_PFC_SOURCE_EXECUTION',result_status=report['status'],
            source_execution_complete=True,case_id=case_id,scientific_fidelity_qualified=False,production_episodes=0,
            files=[self.file_row(root,name) for name in names]))
        return source

    def rebind(self, source):
        report=json.loads((source/'report.json').read_text())
        report['outputs']={name:self.file_row(source,name) for name in report['outputs']}
        self.write_json(source/'report.json',report)
        delivery=json.loads((source.parent/'delivery.json').read_text())
        names={row['path'] for row in delivery['files']} | {'source_run/'+name for name in report['outputs']}
        delivery['files']=[self.file_row(source.parent,name) for name in sorted(names)]
        self.write_json(source.parent/'delivery.json',delivery)

    def add_current(self, source):
        report=json.loads((source/'report.json').read_text());case=report['selected_case']
        values=np.zeros((3,16),dtype=np.float64)
        values[1,4:7]=case['stimulus_strength_nA']*1e-9
        # Native float times: .1 + .2 exceeds .300, so removal occurs at .301 s.
        np.savez(source/'applied_current_observation.npz',source_unit_indices=np.arange(16,dtype=np.int64),
            archive_unit_indices=np.arange(16,dtype=np.int64),segment_start_ticks=np.array([0,2000,6020],dtype=np.int64),
            segment_stop_ticks=np.array([2000,6020,100000],dtype=np.int64),cue_active=np.array([False,True,False]),
            distractor_active=np.zeros(3,dtype=np.bool_),current_A=values)
        self.write_json(source/'applied_current_metadata.json',dict(schema='pfc_applied_current_segments_v1',
            variable='I_stim',population='E',units='A',observed_unit_count=16,dt_seconds=5e-5,duration_ticks=100000,
            stimulus_operation_stride_ticks=20,timing='after_source_stimulus_operation_before_rk2',
            capture_rule='initial_source_operation_and_cue_or_distractor_active_state_transitions',
            source_indices='all_E_source_indices_ascending',archive_indices='E_population_archive_indices_in_E_then_I_unit_order',
            segment_bounds='start_inclusive_stop_exclusive',
            segment_interval_semantics='held by source current-update semantics; not repeated measurements',
            coverage='full E I_stim vectors at segment starts; excludes inhibitory, synaptic and total membrane currents'))
        metadata=json.loads((source/'unit_metadata.json').read_text())
        metadata.update(applied_current_observation_file='applied_current_observation.npz',
                        applied_current_metadata_file='applied_current_metadata.json')
        self.write_json(source/'unit_metadata.json',metadata)
        report['outputs'].update({name:{} for name in analysis._SOURCE_CURRENT_OUTPUTS})
        report['validations']['applied_current_schema']=dict(passed=True)
        report['costs']=dict(applied_current_capture_seconds=.001)
        self.write_json(source/'report.json',report);self.rebind(source)
        return source

    def parameterize_archive(self, source, *, covered_kernel=None):
        report = json.loads((source/'report.json').read_text())
        request = json.loads((source/'request.json').read_text())
        kernel = analysis.kernel_parameters.resolve_kernel_parameters(self.config)
        provenance = analysis.kernel_parameters.circuit_provenance(self.config, analysis.SOURCE_ENTRY_SHA256)
        helper = source.parent/'Libs/population_dynamics/pfc_kernel_parameters.py'
        helper.write_text('# Engineering kernel helper fixture, not source evidence\n')
        helper_sha = sha256_file(helper)
        qualification = report['backend_qualification']
        qroot = source.parent/'qualification'
        receipt = qualification['qualification']
        receipt.update(validated_kernel_parameters=[kernel if covered_kernel is None else covered_kernel],
                       kernel_parameterization_sha256=helper_sha)
        self.write_json(qroot/'comparison/qualification.json', receipt)
        self.write_json(qroot/'delivery.json', dict(status='ARCHIVED_PFC_STREAM_VALIDATION',
            result_status=receipt['status'], files=[self.file_row(qroot, 'comparison/qualification.json')]))
        qualification.update(receipt_sha256=sha256_file(qroot/'comparison/qualification.json'),
                             delivery_sha256=sha256_file(qroot/'delivery.json'))
        for item in (report, request):
            item['code']['kernel_parameterization_sha256'] = helper_sha
            item.update(resolved_kernel_parameters=kernel, circuit_provenance=provenance,
                        backend_qualification=qualification)
        self.write_json(source/'request.json', request)
        self.write_json(source/'report.json', report)
        metadata = json.loads((source/'unit_metadata.json').read_text())
        metadata['coupling'].update(resolved_kernel_parameters=kernel, circuit_provenance=provenance)
        self.write_json(source/'unit_metadata.json', metadata)
        delivery = json.loads((source.parent/'delivery.json').read_text())
        delivery['files'].append(self.file_row(source.parent, str(helper.relative_to(source.parent))))
        self.write_json(source.parent/'delivery.json', delivery)
        self.rebind(source)
        return source

    def test_parameterized_archive_requires_matching_native_tuple_and_helper(self):
        self.config.update(sigma_weight_profile=30., Jpos_excit2excit=1.7)
        self.config_path.write_text(yaml.safe_dump(self.config))
        source = self.parameterize_archive(self.archive('pfc_94001_strong_cue', suffix='_kernel'))
        result = analysis.analyze_source_runs([source], self.config_path, self.root/'kernel_readout')
        self.assertEqual(result['complete_source_episodes'], 1)
        binding = result['source_bindings'][0]['kernel_parameters']
        self.assertEqual(binding['status'], 'EXPLICIT_NATIVE_KERNEL_COVERAGE')
        self.assertEqual(binding['resolved_kernel_parameters']['sigma_weight_profile'], 30.)
        wrong = self.parameterize_archive(self.archive('pfc_94001_strong_cue', suffix='_wrong'),
                                         covered_kernel=dict(sigma_weight_profile=20., Jpos_excit2excit=1.6))
        with self.assertRaisesRegex(ValueError, 'KERNEL_NOT_QUALIFIED'):
            analysis.analyze_source_runs([wrong], self.config_path, self.root/'wrong_kernel')
        with (source.parent/'Libs/population_dynamics/pfc_kernel_parameters.py').open('a') as stream:
            stream.write('# changed\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            analysis.analyze_source_runs([source], self.config_path, self.root/'changed_helper')

    def test_legacy_default_kernel_does_not_authorize_new_weights_or_executables(self):
        source = self.archive('pfc_94001_strong_cue', suffix='_legacy')
        result = analysis.analyze_source_runs([source], self.config_path, self.root/'legacy_kernel')
        self.assertEqual(result['source_bindings'][0]['kernel_parameters']['status'], 'VERIFIED_LEGACY_DEFAULT_ONLY')
        self.config['sigma_weight_profile'] = 30.
        self.config_path.write_text(yaml.safe_dump(self.config))
        nondefault = self.archive('pfc_94001_strong_cue', suffix='_legacy_nondefault')
        with self.assertRaisesRegex(ValueError, 'legacy source archive supports only'):
            analysis.analyze_source_runs([nondefault], self.config_path, self.root/'legacy_rejected')
        with self.assertRaisesRegex(ValueError, 'legacy source archive supports only'):
            analysis._source_kernel_binding(dict(self.config, sigma_weight_profile=20.),
                dict(executor_sha256='unverified_executable'), {}, {}, {})

    def replace_runner_fixture(self, source, text):
        path=source.parent/'Libs/population_dynamics/pfc_source_execution.py'
        path.write_text('# Engineering wrapper fixture: '+text+'\\n')
        digest=sha256_file(path)
        for name in ('report.json','request.json'):
            value=json.loads((source/name).read_text())
            value['code']['runner_sha256']=digest
            self.write_json(source/name,value)
        self.rebind(source)
        return digest

    def test_reviewed_clock_admission_pair_preserves_actual_producer_provenance(self):
        first=self.archive('pfc_94001_strong_cue',suffix='_legacy_wrapper')
        second=self.archive('pfc_94002_strong_cue',suffix='_clock_gate_wrapper')
        old=self.replace_runner_fixture(first,'legacy')
        new=self.replace_runner_fixture(second,'explicit clock gate')
        with mock.patch.object(analysis,'_SOURCE_CLOCK_ADMISSION_WRAPPERS',frozenset((old,new))):
            result=analysis.analyze_source_runs([first,second],self.config_path,self.root/'compatible')
        self.assertEqual(result['complete_source_episodes'],2)
        self.assertEqual(result['registered_source_cases'],9)
        self.assertEqual([row['source_runner_sha256'] for row in result['source_bindings']],[old,new])
        common=result['common_execution_binding']['code']
        self.assertIsNone(common['runner_sha256'])
        self.assertEqual(common['reviewed_runner_compatibility']['allowed_runner_sha256'],sorted((old,new)))
        self.assertFalse(result['scientific_fidelity_qualified'])
        self.assertEqual(result['production_episodes'],0)

    def test_runner_compatibility_does_not_admit_unknown_versions_or_other_clocks(self):
        first=self.archive('pfc_94001_strong_cue',suffix='_known_wrapper')
        second=self.archive('pfc_94002_strong_cue',suffix='_unknown_wrapper')
        old=self.replace_runner_fixture(first,'known')
        unknown=self.replace_runner_fixture(second,'unreviewed numerical change')
        with mock.patch.object(analysis,'_SOURCE_CLOCK_ADMISSION_WRAPPERS',frozenset((old,'different'))):
            with self.assertRaisesRegex(ValueError,'source runs differ'):
                analysis.analyze_source_runs([first,second],self.config_path,self.root/'rejected_version')
        codes=[dict(runner_sha256=value,executor_sha256='same_engine',drive_sha256='same_driver')
               for value in (old,unknown)]
        with mock.patch.object(analysis,'_SOURCE_CLOCK_ADMISSION_WRAPPERS',frozenset((old,unknown))):
            for dt in (.025,.0125):
                self.assertNotEqual(analysis._source_comparison_code(codes[0],dt),
                                    analysis._source_comparison_code(codes[1],dt))
        self.assertEqual(codes[0]['runner_sha256'],old)

    def test_refined_clock_cannot_use_coarse_native_qualification(self):
        self.config['dt_ms']=.025
        self.config_path.write_text(yaml.safe_dump(self.config))
        _,self.registered,self.seeds,self.dimensions=analysis._source_registration(self.config_path)
        source=self.archive('pfc_94001_strong_cue',suffix='_unqualified_clock')
        status,header=analysis._source_header(source,self.config,self.registered,sha256_file(self.config_path))
        self.assertEqual(status['status'],'COMPLETE')
        with self.assertRaisesRegex(ValueError,'source clock not covered'):
            analysis._source_values(header,self.config,self.registered,self.dimensions,sha256_file(self.config_path))

    def test_current_observation_is_verified_without_changing_legacy_readouts(self):
        old=self.archive('pfc_94001_strong_cue',suffix='_old')
        new=self.add_current(self.archive('pfc_94001_strong_cue',suffix='_current'))
        before=analysis.analyze_source_runs([old],self.config_path,self.root/'before')
        after=analysis.analyze_source_runs([new],self.config_path,self.root/'after')
        self.assertEqual((self.root/'before/responses.csv').read_bytes(),(self.root/'after/responses.csv').read_bytes())
        self.assertEqual(before['source_bindings'][0]['applied_current_observation']['status'],'UNAVAILABLE_LEGACY_ARCHIVE')
        observed=after['source_bindings'][0]['applied_current_observation']
        self.assertEqual(observed['status'],'VERIFIED_ACTUAL_CURRENT');self.assertEqual(observed['observed_scalar_values'],48)
        self.assertEqual([row['start_tick'] for row in observed['source_intervals']],[0,2000,6020])
        self.assertEqual([row['nonzero_E_units'] for row in observed['source_intervals']],[0,3,0])
        self.assertEqual(observed['source_intervals'][1]['maximum_current_A'],5e-10)
        self.assertEqual(after['registered_source_cases'],9);self.assertEqual(after['complete_source_episodes'],1)

    def test_current_rejects_corrupted_bound_file_and_incomplete_inventory(self):
        corrupt=self.add_current(self.archive('pfc_94001_cue',suffix='_corrupt'))
        with (corrupt/'applied_current_observation.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            analysis.analyze_source_runs([corrupt],self.config_path,self.root/'corrupt_output')
        for index,names in enumerate([{'applied_current_observation.npz'},{'applied_current_metadata.json'},analysis._SOURCE_CURRENT_OUTPUTS]):
            with self.subTest(missing=names):
                source=self.add_current(self.archive('pfc_94001_cue',suffix='_missing'+str(index)))
                report=json.loads((source/'report.json').read_text())
                for name in names:del report['outputs'][name]
                self.write_json(source/'report.json',report);self.rebind(source)
                with self.assertRaises(ValueError):
                    analysis.analyze_source_runs([source],self.config_path,self.root/('missing_output'+str(index)))

    def test_current_rejects_rebound_identity_timing_and_amplitude_errors(self):
        changes=[('source_unit_indices',lambda a:a.__setitem__(0,1)),
            ('current_A',lambda a:a.__setitem__((1,5),1e-9)),
            ('current_A',lambda a:a.__setitem__((2,5),1e-9)),
            ('current_A',lambda a:a.__setitem__((1,5),np.nan)),
            ('segment_start_ticks',lambda a:a.__setitem__(2,6000)),
            ('segment_stop_ticks',lambda a:a.__setitem__(0,1999)),
            ('cue_active',lambda a:a.__setitem__(1,False))]
        for index,(key,change) in enumerate(changes):
            with self.subTest(key=key,index=index):
                source=self.add_current(self.archive('pfc_94001_strong_cue',suffix='_invalid'+str(index)))
                self.modify_array(source,'applied_current_observation.npz',key,change)
                with self.assertRaisesRegex(ValueError,'applied current'):
                    analysis.analyze_source_runs([source],self.config_path,self.root/('invalid_output'+str(index)))
        source=self.add_current(self.archive('pfc_94001_strong_cue',suffix='_terminal'))
        self.modify_array(source,'terminal_state.npz','E__I_stim',lambda a:a.__setitem__(0,1e-9))
        with self.assertRaisesRegex(ValueError,'applied current terminal'):
            analysis.analyze_source_runs([source],self.config_path,self.root/'terminal_output')

    def test_current_rejects_rebound_units_and_metadata_links(self):
        for index,(name,key,value) in enumerate([
            ('applied_current_metadata.json','units','nA'),
            ('unit_metadata.json','applied_current_observation_file','missing.npz')]):
            with self.subTest(name=name):
                source=self.add_current(self.archive('pfc_94001_cue',suffix='_metadata'+str(index)))
                metadata=json.loads((source/name).read_text());metadata[key]=value
                self.write_json(source/name,metadata);self.rebind(source)
                with self.assertRaisesRegex(ValueError,'applied current'):
                    analysis.analyze_source_runs([source],self.config_path,self.root/('metadata_output'+str(index)))

    def modify_array(self,source,name,key,change):
        with np.load(source/name,allow_pickle=False) as stored:
            arrays={k:stored[k] for k in stored.files}
        change(arrays[key]);np.savez(source/name,**arrays)
        report=json.loads((source/'report.json').read_text())
        report['outputs'][name]=self.file_row(source,name);self.write_json(source/'report.json',report)
        delivery=json.loads((source.parent/'delivery.json').read_text())
        delivery['files']=[self.file_row(source.parent,row['path']) for row in delivery['files']]
        self.write_json(source.parent/'delivery.json',delivery)

    @staticmethod
    def table(path):
        with path.open(newline='') as stream:return list(csv.DictReader(stream))

    def test_partial_retains_registered_denominator_and_full_unit_vectors(self):
        source=self.archive('pfc_94001_strong_cue');out=self.root/'partial'
        report=analysis.analyze_source_runs([source],self.config_path,out)
        self.assertEqual(report['status'],'PARTIAL_SOURCE_EXECUTION_DIAGNOSTIC')
        self.assertEqual(report['registered_source_cases'],9)
        self.assertEqual(report['phase_unit_count_values_verified'],100)
        self.assertEqual(len(self.table(out/'responses.csv')),5)
        self.assertEqual(len(self.table(out/'case_coverage.csv')),9)
        self.assertFalse((out/'paired_effects.csv').exists())
        with np.load(out/'phase_unit_counts.npz') as stored:
            self.assertEqual(len(stored.files),5)
            self.assertTrue(all(stored[k].shape==(20,) and stored[k][-1]==0 for k in stored.files))
        self.assertFalse(report['scientific_fidelity_qualified']);self.assertEqual(report['production_episodes'],0)

    def test_complete_cases_have_independently_replayed_paired_effects(self):
        sources=[self.archive(case_id) for case_id in self.registered];out=self.root/'complete'
        report=analysis.analyze_source_runs(sources,self.config_path,out)
        self.assertEqual(report['status'],'DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC')
        self.assertEqual(report['source_prefix_and_rng_pairs_verified'],9)
        self.assertEqual(report['phase_unit_count_values_verified'],900)
        self.assertEqual(len(self.table(out/'paired_effects.csv')),45)
        self.assertEqual(len(self.table(out/'responses.csv')),45)
        self.assertTrue(report['paired_effects_available'])

    def test_one_missing_case_cannot_produce_pairs(self):
        sources=[self.archive(case_id) for case_id in list(self.registered)[:-1]]
        out=self.root/'missing';report=analysis.analyze_source_runs(sources+[self.root/'unfinished'],self.config_path,out)
        self.assertEqual(report['complete_source_episodes'],8)
        self.assertEqual(len(report['missing_source_cases']),1)
        self.assertEqual(len(report['unresolved_input_paths']),1)
        self.assertFalse((out/'paired_effects.csv').exists())

    def test_all_failed_cases_keep_all_nine_rows_without_reading_arrays(self):
        sources=[self.archive(case_id,failed=True) for case_id in self.registered];out=self.root/'failed_sources'
        report=analysis.analyze_source_runs(sources,self.config_path,out)
        self.assertEqual(report['status'],'BLOCKED_INCOMPLETE_INPUTS')
        self.assertEqual(report['failed_source_attempts'],9)
        self.assertEqual(report['complete_source_episodes'],0)
        self.assertEqual(len(self.table(out/'case_coverage.csv')),9)
        self.assertEqual(self.table(out/'responses.csv'),[])
        self.assertEqual(json.loads((out/'readout_replay.json').read_text())['status'],'NOT_RUN_NO_COMPLETE_CASES')

    def test_rejects_wrong_full_unit_identity_and_counts_after_rebinding(self):
        for name,key in (('units.npz','unit_ids'),('raw.npz','native_spike_counts')):
            with self.subTest(name=name):
                source=self.archive('pfc_94001_cue',suffix=name)
                self.modify_array(source,name,key,lambda a:a.__setitem__(0,int(a[0])+1))
                out=self.root/('rejected_'+name)
                with self.assertRaises(ValueError):analysis.analyze_source_runs([source],self.config_path,out)
                self.assertEqual(json.loads((out/'analysis.json').read_text())['status'],'FAILED')
                self.assertFalse((out/'paired_effects.csv').exists())

    def test_rejects_changed_delivered_output_hash(self):
        source=self.archive('pfc_94001_cue');out=self.root/'hash_rejected'
        with (source/'raw.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            analysis.analyze_source_runs([source],self.config_path,out)

    def test_duplicate_complete_case_is_preserved_and_rejected(self):
        source=self.archive('pfc_94001_cue');out=self.root/'duplicate'
        with self.assertRaisesRegex(ValueError,'duplicate completed'):
            analysis.analyze_source_runs([source,source],self.config_path,out)
        coverage=self.table(out/'case_coverage.csv')
        self.assertEqual(next(row for row in coverage if row['case_id']=='pfc_94001_cue')['status'],'DUPLICATE_COMPLETE')
        self.assertEqual(len(self.table(out/'input_status.csv')),2)

    def test_unequal_final_rng_blocks_paired_effects(self):
        sources=[self.archive(case_id) for case_id in self.registered]
        source=sources[0]
        self.modify_array(source,'final_rng.npz','keys',lambda a:a.__setitem__(0,int(a[0])+1))
        out=self.root/'rng_rejected'
        with self.assertRaisesRegex(ValueError,'matched source RNG'):
            analysis.analyze_source_runs(sources,self.config_path,out)
        self.assertFalse((out/'paired_effects.csv').exists())
        self.assertTrue((out/'source_pairing_checks.csv').exists())

    def test_cli_required_cohort_rejects_partial_without_losing_diagnostics(self):
        # A real subprocess cannot inherit the in-process legacy-fixture allowlist.
        # Explicit bound kernel fixtures exercise CLI coverage without that mock.
        sources=[self.parameterize_archive(self.archive(case_id)) for case_id in self.registered]
        for name,selected,required,expected in (
                ('partial_allowed',sources[:-1],False,0),
                ('partial_required',sources[:-1],True,1),
                ('complete_required',sources,True,0),
                ('unresolved_required',sources+[self.root/'unfinished'],True,1)):
            with self.subTest(name=name):
                out=self.root/name
                command=[sys.executable,'-m',analysis.__name__,
                         '--config',str(self.config_path),'--output',str(out)]
                if required:command.append('--require-complete')
                for source in selected:command.extend(('--source-run',str(source)))
                run=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(run.returncode,expected,run.stderr)
                report=json.loads((out/'analysis.json').read_text())
                self.assertEqual(report['registered_source_cases'],9)
                self.assertEqual(report['complete_source_episodes'],8 if 'partial' in name else 9)
                self.assertEqual(len(self.table(out/'case_coverage.csv')),9)
                self.assertFalse(report['scientific_fidelity_qualified'])
                if 'partial' in name:
                    self.assertEqual(report['status'],'PARTIAL_SOURCE_EXECUTION_DIAGNOSTIC')
                    self.assertEqual(report['missing_source_cases'],[list(self.registered)[-1]])
                elif name=='unresolved_required':
                    self.assertEqual(len(report['unresolved_input_paths']),1)
                else:
                    self.assertEqual(report['status'],'DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC')

    def test_cli_requires_canonical_config_and_reports_zero_coverage(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            analysis.main(['--source-run',str(self.root/'unfinished'),'--output',str(self.root/'bad_cli')])
        self.assertEqual(error.exception.code,2)
        with contextlib.redirect_stdout(io.StringIO()):
            code=analysis.main(['--config',str(self.config_path),'--output',str(self.root/'zero')])
        self.assertEqual(code,1)
        self.assertEqual(len(self.table(self.root/'zero/case_coverage.csv')),9)


if __name__=='__main__':
    unittest.main()
