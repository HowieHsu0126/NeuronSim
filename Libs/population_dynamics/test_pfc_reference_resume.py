"""Crash recovery fixtures exercise storage/RNG/control contracts, not neural fidelity."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from . import pfc_reference as pfc
from .pfc_initial_rng import InitialVoltageTrace
from .pfc_reference_replay import compare as replay_source
from .provenance import sha256_file


class SourceFixture:
    def __init__(self, fail_at=None):
        self.calls = []
        self.seeds = []
        self.fail_at = fail_at
        self.brian = SimpleNamespace(ms=.001, second=1., namp=1e-9, mV=.001,
            defaultclock=SimpleNamespace(t=0., dt=.001),
            prefs=SimpleNamespace(codegen=SimpleNamespace(target=None)),
            start_scope=lambda:None, seed=self.seeds.append, __version__='fixture')

    def simulate_wm(self, **settings):
        self.calls.append(settings)
        if len(self.calls) == self.fail_at:
            raise RuntimeError('fixture interrupted before next complete episode')
        if hasattr(self, 'initial_trace'):
            self.initial_trace.uniform(-60., high=-50., size=settings['N_inhibitory'])
            self.initial_trace.uniform(-60., high=-50., size=settings['N_excitatory'])
        self.brian.defaultclock.t = settings['sim_time']
        e = SimpleNamespace(i=np.array([0, 2]), t=np.array([.001, .005]),
                            count=np.array([1, 0, 1, 0]), source=SimpleNamespace(v=np.full(4, -.065)))
        i = SimpleNamespace(i=np.array([1]), t=np.array([.003]),
                            count=np.array([0, 1]), source=SimpleNamespace(v=np.full(2, -.064)))
        return None, e, None, None, None, i


class PFCResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.revision = 'fixture_source_revision'
        (self.source/'source.json').write_text(json.dumps(dict(revision=self.revision)))
        self.config = dict(scope='source_reproduction',expected_source_revision=self.revision,
            excitatory_neurons=4,inhibitory_neurons=2,weight_scaling_factor=1.,dt_ms=1.,
            duration_ms=20.,excitatory_preference_bins=2,readout_bin_ms=2.,
            stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_strength_nA=.06,
            stimulus_start_ms=2.,stimulus_duration_ms=2.,distractor_center_deg=90.,
            distractor_width_deg=40.,distractor_start_ms=6.,distractor_duration_ms=2.,
            state_monitored_per_population=2,source_rng_seed=101,
            cases=[dict(case_id=name,source_rng_seed=seed,stimulus_strength_nA=current,distractor_strength_nA=0.)
                   for name,seed,current in [('a',101,.06),('b',102,0.),('c',103,.5)]])
        self.config_path = self.root/'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.out = self.root/'interrupted'

    def install_fixture_initial_rng(self, module, seed):
        module.initial_trace=InitialVoltageTrace(seed)
        return module.initial_trace

    def independent_config(self):
        self.config.pop('source_rng_seed')
        self.config['random_stream_policy']='independent_initial_external'
        for row in self.config['cases']:
            old=row.pop('source_rng_seed')
            row.update(initial_rng_seed=1000+old,external_noise_seed=2000+old)
        self.config_path.write_text(yaml.safe_dump(self.config))

    def run_source(self, output=None, *, resume=False, fail_at=None):
        fixture = SourceFixture(fail_at)
        self.last_fixture = fixture
        with patch.object(pfc, 'b2', fixture.brian), patch.object(
                pfc, 'load_source', return_value=(fixture, dict(revision=self.revision), dict(sha256='e'*64))), patch.object(
                pfc, 'install_initial_voltage_rng', side_effect=self.install_fixture_initial_rng):
            report = pfc.execute(self.source, self.config_path, output or self.out, resume=resume)
        return report, fixture

    def interrupt(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
            self.run_source(fail_at=2)
        self.assertFalse((self.out/'report.json').exists())
        catalog = json.loads((self.out/'dataset/dataset.json').read_text())
        self.assertEqual(catalog['complete_episodes'], 1)
        self.assertEqual(len(catalog['shards']), 1)
        return catalog

    def rewrite_shard_metadata(self, callback):
        catalog_path = self.out/'dataset/dataset.json'
        catalog = json.loads(catalog_path.read_text())
        shard = self.out/'dataset'/catalog['shards'][0]['path']
        with h5py.File(shard, 'r+') as stream:
            callback(stream['a'])
        catalog['shards'][0]['sha256'] = sha256_file(shard)
        catalog_path.write_text(json.dumps(catalog))

    def test_resume_skips_committed_episode_and_matches_fresh_output(self):
        self.interrupt()
        raw_hash = sha256_file(self.out/'a_native.npz')
        shard_hash = sha256_file(self.out/'dataset/episodes_000000.h5')
        report, fixture = self.run_source(resume=True)
        self.assertEqual(fixture.seeds, [102, 103])
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(report['reused_complete_case_ids'], ['a'])
        self.assertTrue(report['resumed'])
        self.assertEqual([row['case_id'] for row in report['cases']], ['a', 'b', 'c'])
        self.assertEqual(sha256_file(self.out/'a_native.npz'), raw_hash)
        self.assertEqual(sha256_file(self.out/'dataset/episodes_000000.h5'), shard_hash)
        self.assertEqual(json.loads((self.out/'dataset/dataset.json').read_text())['complete_episodes'], 3)
        fresh, fresh_fixture = self.run_source(self.root/'fresh')
        self.assertEqual(fresh_fixture.seeds, [101, 102, 103])
        self.assertEqual(report['independently_replayed_events'], fresh['independently_replayed_events'])
        for name in ['a', 'b', 'c']:
            with np.load(self.out/(name+'_native.npz')) as recovered, np.load(self.root/'fresh'/(name+'_native.npz')) as original:
                self.assertEqual(recovered.files, original.files)
                for key in original.files:
                    np.testing.assert_array_equal(recovered[key], original[key])

    def test_closed_shard_catalog_gap_uses_existing_recovery(self):
        catalog = self.interrupt()
        catalog['shards'] = []
        catalog['complete_episodes'] = 0
        (self.out/'dataset/dataset.json').write_text(json.dumps(catalog))
        report, fixture = self.run_source(resume=True)
        self.assertEqual(report['reused_complete_case_ids'], ['a'])
        self.assertEqual(fixture.seeds, [102, 103])
        catalog = json.loads((self.out/'dataset/dataset.json').read_text())
        self.assertEqual(catalog['complete_episodes'], 3)
        self.assertEqual(len(catalog['shards']), 3)

    def test_default_existing_output_rejection_and_explicit_resume_requirement(self):
        self.interrupt()
        before = sha256_file(self.out/'dataset/dataset.json')
        with self.assertRaisesRegex(FileExistsError, 'OUTPUT_EXISTS'):
            self.run_source()
        self.assertEqual(sha256_file(self.out/'dataset/dataset.json'), before)
        with self.assertRaisesRegex(FileNotFoundError, 'RESUME_OUTPUT_REQUIRED'):
            self.run_source(self.root/'missing', resume=True)

    def test_completed_output_cannot_be_resumed_or_overwritten(self):
        self.run_source()
        before = sha256_file(self.out/'report.json')
        with self.assertRaisesRegex(FileExistsError, 'ALREADY_COMPLETE'):
            self.run_source(resume=True)
        self.assertEqual(sha256_file(self.out/'report.json'), before)

    def test_changed_config_rejected_before_archive_mutation(self):
        self.interrupt()
        before = sha256_file(self.out/'dataset/dataset.json')
        self.config['stimulus_center_deg'] = 121.
        self.config_path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, 'RESUME_BINDING_CHANGED'):
            self.run_source(resume=True)
        self.assertEqual(sha256_file(self.out/'dataset/dataset.json'), before)
        self.assertEqual(self.last_fixture.calls, [])

    def test_corrupt_raw_rejected_without_simulation(self):
        self.interrupt()
        with (self.out/'a_native.npz').open('ab') as stream:
            stream.write(b'corruption')
        before = sha256_file(self.out/'dataset/dataset.json')
        with self.assertRaisesRegex(ValueError, 'RAW_DIGEST_CHANGED'):
            self.run_source(resume=True)
        self.assertEqual(sha256_file(self.out/'dataset/dataset.json'), before)
        self.assertEqual(self.last_fixture.calls, [])

    def test_corrupt_committed_hdf5_rejected_without_mutation(self):
        self.interrupt()
        with (self.out/'dataset/episodes_000000.h5').open('ab') as stream:
            stream.write(b'corruption')
        before = sha256_file(self.out/'dataset/dataset.json')
        with self.assertRaisesRegex(ValueError, 'ARCHIVE_CONTENT_CHANGED'):
            self.run_source(resume=True)
        self.assertEqual(sha256_file(self.out/'dataset/dataset.json'), before)

    def test_untracked_raw_and_corrupt_writing_shard_are_preserved(self):
        self.interrupt()
        raw = self.out/'b_native.npz'
        partial = self.out/'dataset/episodes_000001.h5.writing'
        raw.write_bytes(b'untracked')
        partial.write_bytes(b'corrupt interrupted shard')
        before = {path:sha256_file(path) for path in (raw, partial, self.out/'dataset/dataset.json')}
        with self.assertRaisesRegex(ValueError, 'UNTRACKED_OR_MISSING_RAW'):
            self.run_source(resume=True)
        self.assertEqual({path:sha256_file(path) for path in before}, before)
        self.assertEqual(self.last_fixture.calls, [])

    def test_changed_source_rng_identity_rejected_even_with_updated_shard_digest(self):
        self.interrupt()
        def change(group):
            metadata = json.loads(group.attrs['metadata_json'])
            metadata['seeds']['source_rng_seed'] = 999
            group.attrs['metadata_json'] = json.dumps(metadata)
        self.rewrite_shard_metadata(change)
        with self.assertRaisesRegex(ValueError, 'EPISODE_IDENTITY_CHANGED'):
            self.run_source(resume=True)

    def test_changed_control_clock_rejected_even_with_updated_shard_digest(self):
        self.interrupt()
        def change(group):
            ticks = group['control_ticks'][:]
            ticks[1] += 1
            group['control_ticks'][:] = ticks
        self.rewrite_shard_metadata(change)
        with self.assertRaisesRegex(ValueError, 'CONTROL_PROGRAM_CHANGED'):
            self.run_source(resume=True)

    def test_raw_hdf5_disagreement_rejected_after_digest_rebinding(self):
        self.interrupt()
        raw = self.out/'a_native.npz'
        with np.load(raw, allow_pickle=False) as stream:
            arrays = {name:stream[name] for name in stream.files}
        arrays['spike_ticks'][0] = 2
        arrays['spike_seconds'][0] = .002
        np.savez_compressed(raw, **arrays)
        def change(group):
            metadata = json.loads(group.attrs['metadata_json'])
            metadata['source_record']['raw_sha256'] = sha256_file(raw)
            group.attrs['metadata_json'] = json.dumps(metadata)
        self.rewrite_shard_metadata(change)
        with self.assertRaisesRegex(ValueError, 'RAW_EPISODE_MISMATCH'):
            self.run_source(resume=True)
        self.assertEqual(self.last_fixture.calls, [])

    def test_incomplete_duration_receipt_rejected(self):
        self.interrupt()
        def change(group):
            metadata = json.loads(group.attrs['metadata_json'])
            metadata['source_record']['recorded_until_tick'] -= 1
            group.attrs['metadata_json'] = json.dumps(metadata)
        self.rewrite_shard_metadata(change)
        with self.assertRaisesRegex(ValueError, 'COMPLETION_NOT_ESTABLISHED'):
            self.run_source(resume=True)


    def run_profiled_source(self, *, fail_at=None):
        fixture = SourceFixture(fail_at)
        with patch.object(pfc, 'b2', fixture.brian), patch.object(
                pfc, 'load_source', return_value=(fixture, dict(revision=self.revision), dict(sha256='e'*64))), patch.object(
                pfc, 'install_initial_voltage_rng', side_effect=self.install_fixture_initial_rng):
            return pfc.profiled_execute(self.source, self.config_path, self.out,
                                        self.root/'execution_profile.json')

    def test_profile_retains_source_values_and_records_full_entrypoint(self):
        report = self.run_profiled_source()
        profile = json.loads((self.root/'execution_profile.json').read_text())
        self.assertEqual(profile['status'], 'PROFILED_SOURCE_EXECUTION_COMPLETE')
        self.assertEqual(profile['report_sha256'], sha256_file(self.out/'report.json'))
        self.assertGreater(profile['process_peak_rss_bytes'], 0)
        self.assertTrue(profile['profiler_overhead_included'])
        self.assertFalse(profile['independent_cost_comparison_qualified'])
        self.assertTrue(report['python_profiler_active'])
        self.assertTrue(any(row['function']=='simulate_wm' and row['total_calls']==3
                            for row in profile['functions']))
        self.assertTrue(any(row['function']=='window' for row in profile['functions']))
        fresh, _ = self.run_source(self.root/'fresh')
        self.assertFalse(fresh['python_profiler_active'])
        for name in ['a', 'b', 'c']:
            with np.load(self.out/(name+'_native.npz')) as profiled, np.load(self.root/'fresh'/(name+'_native.npz')) as plain:
                self.assertEqual(profiled.files, plain.files)
                for key in plain.files:
                    np.testing.assert_array_equal(profiled[key], plain[key])
        before = sha256_file(self.root/'execution_profile.json')
        with self.assertRaisesRegex(FileExistsError, 'PROFILE_EXISTS'):
            self.run_profiled_source()
        self.assertEqual(sha256_file(self.root/'execution_profile.json'), before)

    def test_profile_failure_propagates_and_preserves_completed_episode(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
            self.run_profiled_source(fail_at=2)
        profile = json.loads((self.root/'execution_profile.json').read_text())
        self.assertEqual(profile['status'], 'PROFILED_SOURCE_EXECUTION_FAILED')
        self.assertEqual(profile['failure']['error_type'], 'RuntimeError')
        self.assertIsNone(profile['report_sha256'])
        self.assertFalse((self.out/'report.json').exists())
        catalog = json.loads((self.out/'dataset/dataset.json').read_text())
        self.assertEqual(catalog['complete_episodes'], 1)
        self.assertIsNone(pfc.sys.getprofile())

    def test_profile_cannot_precreate_source_output(self):
        with self.assertRaisesRegex(ValueError, 'OUTSIDE_SOURCE_OUTPUT'):
            pfc.profiled_execute(self.source, self.config_path, self.out,
                                 self.out/'execution_profile.json')
        self.assertFalse(self.out.exists())

    def test_100k_config_preserves_source_scaling_duration_and_three_blocks(self):
        config = yaml.safe_load((Path(__file__).parents[1]/'configs/neural_population_pfc_100k_source.yaml').read_text())
        self.assertEqual(config['excitatory_neurons']+config['inhibitory_neurons'], 100000)
        self.assertAlmostEqual(config['excitatory_neurons']*config['weight_scaling_factor'], 2048.)
        self.assertAlmostEqual(config['inhibitory_neurons']*config['weight_scaling_factor'], 512.)
        self.assertEqual(config['duration_ms'], 5000.)
        self.assertEqual(config['dt_ms'], .05)
        self.assertEqual(config['synapse_execution'], 'factorized_uniform_projections')
        cases=pfc.case_settings(config)
        self.assertEqual({row['source_rng_seed'] for row in cases}, {94001,94002,94003})
        self.assertEqual(len(cases), 9)
        for seed in {94001,94002,94003}:
            self.assertEqual({row['stimulus_strength_nA'] for row in cases if row['source_rng_seed']==seed}, {0.,.06,.5})


    def test_independent_initial_states_are_archived_and_resumed(self):
        self.independent_config();self.interrupt()
        initial=self.out/'a_initial_voltage.npy';before=sha256_file(initial)
        report,fixture=self.run_source(resume=True)
        self.assertEqual(fixture.seeds,[2102,2103])
        self.assertEqual(sha256_file(initial),before)
        self.assertIsNone(report['independent_source_rng_blocks'])
        self.assertEqual(report['source_randomization_blocks'],3)
        for name,seed in [('a',1101),('b',1102),('c',1103)]:
            rng=np.random.RandomState(seed)
            inhibitory=rng.uniform(-60.,-50.,size=2);excitatory=rng.uniform(-60.,-50.,size=4)
            np.testing.assert_array_equal(np.load(self.out/(name+'_initial_voltage.npy')),
                                          np.r_[excitatory,inhibitory])
        catalog=json.loads((self.out/'dataset/dataset.json').read_text())
        with h5py.File(self.out/'dataset'/catalog['shards'][0]['path'],'r') as stream:
            metadata=json.loads(stream['a'].attrs['metadata_json'])
        self.assertEqual(metadata['seeds'],dict(initial_rng_seed=1101,external_noise_seed=2101))
        self.assertEqual(metadata['source_record']['initial_voltage_role'],'source_provenance_not_model_input')
        self.assertEqual(metadata['block_id'],'pfc_initial_1101_external_2101')

    def test_corrupt_independent_initial_state_blocks_resume(self):
        self.independent_config();self.interrupt()
        with (self.out/'a_initial_voltage.npy').open('ab') as stream:stream.write(b'corrupt')
        before=sha256_file(self.out/'dataset/dataset.json')
        with self.assertRaisesRegex(ValueError,'INITIAL_VOLTAGE_BINDING_CHANGED'):
            self.run_source(resume=True)
        self.assertEqual(sha256_file(self.out/'dataset/dataset.json'),before)
        self.assertEqual(self.last_fixture.calls,[])

    def test_untracked_initial_state_blocks_resume_without_deleting_it(self):
        self.independent_config();self.interrupt()
        extra=self.out/'unknown_initial_voltage.npy';np.save(extra,np.zeros(6))
        before=sha256_file(extra)
        with self.assertRaisesRegex(ValueError,'UNTRACKED_OR_MISSING_INITIAL_VOLTAGE'):
            self.run_source(resume=True)
        self.assertEqual(sha256_file(extra),before)
        self.assertEqual(self.last_fixture.calls,[])


    def test_independent_initial_replay_checks_rng_meaning_after_rebinding(self):
        self.independent_config()
        self.run_source()
        second=self.root/'second';self.run_source(second)
        report=replay_source(self.out,second,self.root/'replay.json')
        self.assertEqual(report['source_initial_voltage_rng_replay_values'],18)
        self.assertEqual(report['fresh_process_initial_voltage_values_checked'],18)
        for root in [self.out,second]:
            initial=root/'a_initial_voltage.npy';values=np.load(initial);values[0]+=.01
            np.save(initial,values)
            path=root/'report.json';metadata=json.loads(path.read_text())
            metadata['cases'][0]['initial_voltage_sha256']=sha256_file(initial)
            path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'INITIAL_VOLTAGE_RNG_REPLAY_FAILED'):
            replay_source(self.out,second,self.root/'invalid_replay.json')


    def test_changed_initial_adapter_blocks_resume_before_mutation(self):
        self.independent_config();self.interrupt()
        before=sha256_file(self.out/'dataset/dataset.json')
        original=pfc.sha256_file
        def changed(path):
            return 'f'*64 if Path(path).name=='pfc_initial_rng.py' else original(path)
        with patch.object(pfc,'sha256_file',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'RESUME_BINDING_CHANGED'):
                self.run_source(resume=True)
        self.assertEqual(sha256_file(self.out/'dataset/dataset.json'),before)
        self.assertEqual(self.last_fixture.calls,[])


if __name__ == '__main__':
    unittest.main()
