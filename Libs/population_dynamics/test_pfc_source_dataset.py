"""Engineering fixtures for archive-only PFC data conversion; no scientific evidence."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

import h5py
import numpy as np
import yaml

from . import pfc_source_dataset as exporter
from . import test_pfc_source_scale_analysis as fixture_module
from . import test_pfc_distractor_response as distractor_fixture


class PfcSourceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.SourceReadoutTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.config=self.fixture.config_path
        self.fixture.config['excitatory_preference_bins']=4
        self.config.write_text(yaml.safe_dump(self.fixture.config))

    def source(self,case='pfc_94001_strong_cue',suffix='',observed=True):
        source=self.fixture.archive(case,suffix=suffix)
        return self.fixture.add_current(source) if observed else source

    def test_cli_preserves_complete_events_actual_clock_and_privileged_observations(self):
        source=self.source();output=self.root/'export'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(exporter.main(['--source-run',str(source),'--config',str(self.config),
                                           '--output',str(output)]),0)
        report=json.loads((output/'report.json').read_text())
        self.assertEqual(report['status'],'EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET')
        self.assertEqual(report['complete_episodes'],1);self.assertEqual(report['registered_source_cases'],9)
        self.assertEqual(report['production_episodes'],0);self.assertEqual(report['new_source_simulations'],0)
        dataset=exporter.CircuitDataset(output/'dataset',splits=('development',))
        self.assertFalse(dataset.catalog['source_fidelity_qualified'])
        self.assertEqual(dataset.catalog['binding']['scope'],'source_reproduction')
        view=dataset.window('pfc_94001_strong_cue',boundary_tick=4000,history_ticks=4000,horizon_ticks=4000,bin_ticks=100)
        np.testing.assert_array_equal(view['known_controls'][60],[.1,0.,.2,0.])
        self.assertNotIn('privileged',view)
        row=dataset.episodes['pfc_94001_strong_cue'];shard=dataset.shards[row['shard']]
        with h5py.File(dataset._verified(shard['path'],shard['sha256']),'r') as stored:
            episode=stored['pfc_94001_strong_cue']
            np.testing.assert_array_equal(episode['control_ticks'][:],[0,2000,6020])
            np.testing.assert_array_equal(episode['privileged/delivered_current__source_unit_indices'][:],np.arange(16))
            self.assertEqual(episode['privileged/delivered_current__current_A'].shape,(3,16))
            self.assertEqual(episode['privileged/initial_voltage_mV'].shape,(20,))
            self.assertEqual(episode['privileged/terminal__E__v'].shape,(16,))
            metadata=json.loads(episode.attrs['metadata_json'])
            self.assertEqual(metadata['seeds'],{'source_rng_seed':94001})
            self.assertEqual(metadata['split'],'development')

    def benchmark_source(self, suffix=''):
        from .benchmark_tasks import pfc_working_memory_config, task_protocol
        from .circuit_campaign import campaign_config, episode_identities
        from itertools import islice

        campaign = campaign_config(Path(__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml')
        identity = next(islice(episode_identities(campaign, 'pfc_working_memory'), 4, 5))
        self.fixture.config = pfc_working_memory_config(self.fixture.config, task_protocol('pfc_working_memory'), identity)
        self.config.write_text(yaml.safe_dump(self.fixture.config))
        case_id = identity['episode_id']
        self.fixture.registered = {case_id: (identity['background_seed'], 'strong_cue')}
        self.fixture.dimensions = exporter._benchmark_registration(self.fixture.config)[3]
        source = self.fixture.add_current(self.fixture.archive(case_id, suffix=suffix))
        report = json.loads((source/'report.json').read_text())
        seeds = dict(initial_rng_seed=identity['initial_state_seed'], external_noise_seed=identity['background_seed'])
        report.update(seeds=seeds, random_stream_policy='independent_initial_external')
        report['driver'].update(seeds=seeds, random_stream_policy='independent_initial_external')
        self.fixture.write_json(source/'report.json', report)
        from .pfc_native_drive import create_drive
        initial, _ = create_drive(self.fixture.config, self.fixture.config['cases'][0])
        np.save(source/'initial_voltage.npy', initial)
        values = np.zeros((5,16), dtype=np.float64)
        values[1,4:7] = .5e-9; values[3,3:6] = .5e-9
        np.savez(source/'applied_current_observation.npz', source_unit_indices=np.arange(16,dtype=np.int64),
            archive_unit_indices=np.arange(16,dtype=np.int64),
            segment_start_ticks=np.array([0,2000,6020,48000,49000],dtype=np.int64),
            segment_stop_ticks=np.array([2000,6020,48000,49000,100000],dtype=np.int64),
            cue_active=np.array([False,True,False,False,False]),
            distractor_active=np.array([False,False,False,True,False]), current_A=values)
        self.fixture.rebind(source)
        return source, identity

    def test_benchmark_export_preserves_two_streams_and_actual_circuit_identity(self):
        source, identity = self.benchmark_source()
        output = self.root/'benchmark_export'
        report = exporter.export([source], self.config, output)
        assert report['complete_episodes'] == 1 and report['production_episodes'] == 0
        dataset = exporter.CircuitDataset(output/'dataset', splits=('development',))
        row = dataset.episodes[identity['episode_id']]
        shard = dataset.shards[row['shard']]
        with h5py.File(dataset._verified(shard['path'],shard['sha256']), 'r') as stored:
            episode = stored[identity['episode_id']]
            metadata = json.loads(episode.attrs['metadata_json'])
            assert metadata['split'] == 'development'
            assert metadata['seeds'] == dict(initial_rng_seed=210000000, external_noise_seed=220000000)
            assert metadata['block_id'] == 'pfc_initial_210000000_external_220000000'
            assert metadata['circuit_id'] != identity['circuit_id']
            candidate = metadata['source_record']['benchmark_episode_candidate']
            assert candidate['circuit_parameter_domain_bound'] is False
            assert metadata['source_record']['initial_and_background_streams'] == 'independent_initial_voltage_and_external_noise_streams'
            np.testing.assert_array_equal(episode['control_ticks'][:], [0,2000,6020,48000,49000])
            assert metadata['control_features'][1]['center_degree'] == 90.0
            assert metadata['control_features'][1]['width_degree'] == 40.0
        view = dataset.window(identity['episode_id'],boundary_tick=48000,history_ticks=1000,horizon_ticks=1000,bin_ticks=100)
        assert 'privileged' not in view
        np.testing.assert_array_equal(view['known_controls'][10:,1], np.full(10,.5))

    def test_benchmark_parameterized_export_preserves_tuple_and_real_circuit(self):
        source, identity = self.benchmark_source('_parameterized')
        self.fixture.config.update(sigma_weight_profile=30.0, Jpos_excit2excit=1.7)
        self.config.write_text(yaml.safe_dump(self.fixture.config))
        self.fixture.write_json(source/'config.yaml', self.fixture.config)
        # Restore canonical YAML bytes, then bind the explicitly covered fixture.
        (source/'config.yaml').write_bytes(self.config.read_bytes())
        report = json.loads((source/'report.json').read_text())
        report['resolved_config'] = self.fixture.config
        report['config_sha256'] = exporter.sha256_file(self.config)
        request = json.loads((source/'request.json').read_text())
        request['config_sha256'] = report['config_sha256']
        self.fixture.write_json(source/'request.json', request)
        self.fixture.write_json(source/'report.json', report)
        self.fixture.parameterize_archive(source)
        output = self.root/'benchmark_parameter_export'
        result = exporter.export([source], self.config, output)
        dataset = exporter.CircuitDataset(output/'dataset', splits=('development',))
        row = dataset.episodes[identity['episode_id']]
        expected = exporter.circuit_provenance(self.fixture.config, exporter.SOURCE_ENTRY_SHA256)
        assert row['circuit_id'] == 'pfc_'+expected['parameter_identity'][:16]
        assert row['circuit_id'] != identity['circuit_id']
        assert result['production_episodes'] == 0

    def test_benchmark_export_rejects_rebound_seed_or_schedule_and_candidate_promotion(self):
        source, _ = self.benchmark_source('_invalid')
        original = json.loads((source/'report.json').read_text())
        for seed_name in ('initial_rng_seed','external_noise_seed'):
            report = json.loads(json.dumps(original))
            report['seeds'][seed_name] += 1
            self.fixture.write_json(source/'report.json', report); self.fixture.rebind(source)
            with self.assertRaisesRegex(ValueError,'source RNG block'):
                exporter.export([source],self.config,self.root/('bad_'+seed_name))
        self.fixture.write_json(source/'report.json',original); self.fixture.rebind(source)
        for key,value in (('distractor_start_ms',2200.0),('random_stream_policy','source_global')):
            config = dict(self.fixture.config, **{key:value})
            path=self.root/(key+'.yaml');path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'COMPILER_BINDING'):
                exporter.export([source],path,self.root/('changed_'+key))
        config=json.loads(json.dumps(self.fixture.config))
        config['benchmark_episode_candidate']['execution_scope']='production'
        path=self.root/'promoted.yaml';path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError,'COMPILER_BINDING'):
            exporter.export([source],path,self.root/'promoted')

    def test_registered_distractor_cli_preserves_nonzero_control_and_private_current(self):
        fixture=distractor_fixture.DistractorResponseTests()
        original_setup=fixture_module.SourceReadoutTests.setUp
        def with_field_map(instance):
            original_setup(instance)
            instance.config['excitatory_preference_bins']=4
        with mock.patch.object(fixture_module.SourceReadoutTests,'setUp',with_field_map):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        output=fixture.root/'dataset_export'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(exporter.main(['--source-run',str(fixture.candidate.parent),
                '--config',str(fixture.config_path),'--output',str(output),
                '--distractor-baseline',str(fixture.baseline.parent)]),0)
        report=json.loads((output/'report.json').read_text())
        self.assertEqual(report['registered_source_cases'],1)
        self.assertEqual(report['complete_episodes'],1)
        self.assertEqual(report['production_episodes'],0)
        dataset=exporter.CircuitDataset(output/'dataset',splits=('development',))
        view=dataset.window('pfc_94001_strong_cue_distractor',boundary_tick=8000,
            history_ticks=1000,horizon_ticks=1000,bin_ticks=100)
        np.testing.assert_array_equal(view['known_controls'][:10,1],np.zeros(10))
        np.testing.assert_array_equal(view['known_controls'][10:,1],np.full(10,.5))
        np.testing.assert_array_equal(view['known_controls'][10:,3],np.ones(10))
        self.assertNotIn('privileged',view)
        self.assertEqual(view['control_features'][1]['unit'],'nA')
        self.assertFalse(dataset.catalog['source_fidelity_qualified'])

    def test_nondefault_kernel_export_retains_weighted_circuit_identity(self):
        baseline = self.source(suffix='_default_kernel')
        first = self.root/'default_export'
        exporter.export([baseline], self.config, first)
        original = exporter.CircuitDataset(first/'dataset', splits=('development',))
        self.fixture.config.update(sigma_weight_profile=30., Jpos_excit2excit=1.7)
        self.config.write_text(yaml.safe_dump(self.fixture.config))
        nondefault = self.fixture.parameterize_archive(self.source(suffix='_changed_kernel'))
        second = self.root/'parameterized_export'
        exporter.export([nondefault], self.config, second)
        changed = exporter.CircuitDataset(second/'dataset', splits=('development',))
        a = next(iter(original.episodes.values()))
        b = next(iter(changed.episodes.values()))
        self.assertNotEqual(a['circuit_id'], b['circuit_id'])
        expected = exporter.circuit_provenance(self.fixture.config, exporter.SOURCE_ENTRY_SHA256)
        self.assertEqual(b['circuit_id'], 'pfc_'+expected['parameter_identity'][:16])
        self.assertEqual(len(changed.catalog['circuits']), 1)

    def test_source_seed_changes_episode_block_but_not_circuit_identity(self):
        first=self.source('pfc_94001_strong_cue');second=self.source('pfc_94002_strong_cue')
        output=self.root/'two'
        report=exporter.export([first,second],self.config,output)
        dataset=exporter.CircuitDataset(output/'dataset',splits=('development',))
        rows=list(dataset.episodes.values())
        self.assertEqual(report['complete_episodes'],2)
        self.assertEqual(len(dataset.catalog['circuits']),1)
        self.assertEqual(rows[0]['circuit_id'],rows[1]['circuit_id'])
        self.assertNotEqual(rows[0]['block_id'],rows[1]['block_id'])

    def test_staged_conversion_keeps_durable_origin_and_rejects_changed_binding(self):
        source=self.source();staged=self.root/'staged';shutil.copytree(source.parent,staged)
        delivery=exporter.sha256_file(staged/'delivery.json')
        origins=self.root/'input_origins.json'
        rows=[dict(staged=str(staged),archive=str(source.parent),case_id='pfc_94001_strong_cue',delivery_sha256=delivery)]
        origins.write_text(json.dumps(rows));output=self.root/'staged_export'
        exporter.export([staged],self.config,output,input_origins=origins)
        dataset=exporter.CircuitDataset(output/'dataset',splits=('development',))
        self.assertEqual(dataset.catalog['binding']['source_bindings'][0]['path'],str(source))
        for durable in (self.root/'nonexistent', self.root/'different_receipt'):
            with self.subTest(durable=durable.name):
                if durable.name=='different_receipt':
                    durable.mkdir();(durable/'delivery.json').write_text('{}')
                rows[0]['archive']=str(durable);origins.write_text(json.dumps(rows))
                with self.assertRaisesRegex(ValueError,'DURABLE_ORIGIN_BINDING_MISMATCH'):
                    exporter.export([staged],self.config,self.root/('bad_'+durable.name),input_origins=origins)
        rows[0]['archive']=str(source.parent)
        rows[0]['delivery_sha256']='0'*64;origins.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'STAGED_ORIGIN_BINDING_MISMATCH'):
            exporter.export([staged],self.config,self.root/'changed_origin',input_origins=origins)

    def test_duplicate_source_case_is_not_counted_twice(self):
        first=self.source(suffix='_first');second=self.source(suffix='_other_backend_receipt')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_SOURCE_CASE'):
            exporter.export([first,second],self.config,self.root/'duplicate')
        self.assertFalse((self.root/'duplicate').exists())

    def test_legacy_control_observation_is_not_synthesized(self):
        source=self.source(observed=False);output=self.root/'legacy'
        with self.assertRaisesRegex(ValueError,'OBSERVED_CURRENT_REQUIRED'):
            exporter.export([source],self.config,output)
        report=json.loads((output/'report.json').read_text())
        self.assertEqual(report['status'],'FAILED');self.assertEqual(report['production_episodes'],0)
        self.assertFalse(report['scientific_fidelity_qualified'])

    def test_production_config_and_existing_output_are_rejected(self):
        source=self.source();output=self.root/'existing';output.mkdir()
        with self.assertRaises(FileExistsError):exporter.export([source],self.config,output)
        config=dict(self.fixture.config,scope='production')
        changed=self.root/'production.yaml';changed.write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):exporter.export([source],changed,self.root/'production')
        self.assertFalse((self.root/'production').exists())

    def test_window_count_control_or_privilege_failure_prevents_verified_export(self):
        original=exporter.CircuitDataset.window
        for label in ('counts','controls','privileged'):
            with self.subTest(label=label):
                source=self.source(suffix='_'+label);output=self.root/('bad_'+label)
                def changed(dataset,*args,**kwargs):
                    result=original(dataset,*args,**kwargs)
                    if label=='counts':result['history'][0,0]+=1
                    elif label=='controls':result['known_controls'][:]=0
                    else:result['privileged']={}
                    return result
                with mock.patch.object(exporter.CircuitDataset,'window',changed):
                    with self.assertRaisesRegex(ValueError,'PFC_DATASET_'):
                        exporter.export([source],self.config,output)
                report=json.loads((output/'report.json').read_text())
                self.assertEqual(report['status'],'FAILED');self.assertEqual(report['production_episodes'],0)


if __name__=='__main__':
    unittest.main()
