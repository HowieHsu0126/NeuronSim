"""Synthetic command/archive tests; no source rates, spikes, or network execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import h5py
import numpy as np

from .circuit_data import (SCHEMA as CIRCUIT_SCHEMA, CircuitArchiveWriter, CircuitDataset,
                           CircuitEpisode, CircuitUnits)
from .provenance import freeze_json, sha256_file
from . import v1_visual_controls as visual_controls
from .v1_lgn_spikes import write_spike_trial
from .v1_circuit_stimulus import load_inputs
from .v1_visual_controls import (
    CONSTRUCTED_FAMILY, CONTROL_FEATURES, FRAME_COUNT, SCHEMA, compare_source_movie, load_visual_controls,
    make_commands, reconstruct_frame, validate_constructed_source_bindings, validate_native_binding, validate_source_bindings,
)


def stimulus():
    return dict(duration_s=3.0, gray_screen_s=0.5,
                spatial_frequency_cycles_per_degree=0.04, temporal_frequency_hz=2.0,
                direction_deg=270.0, contrast_percent=80.0)


def source_reports(*, contrast_percent=80.0, include_rate_spike_generation=False):
    config = dict(scope='development_source_runtime_qualification', unit_coverage='all_source_units',
                  converter_python='/usr/bin/python3', stimulus=stimulus())
    config['stimulus']['contrast_percent'] = contrast_percent
    rate = dict(status='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT', config=config,
                source_units=17400, source_unit_ids=list(range(17400)), rate_shape=[17400, 3001],
                time_units='s', firing_rate_units='Hz')
    movie = dict(status='SOURCE_MOVIE_CONTROL_WITNESS_PASS', gray_frames=501,
        first_retained_grating_local_frame=1, source_default_phase=0.0,
        frame_rate_hz=1000.0, movie_shape=[3001, 48, 96], prepared_runtime_sha256='e'*64,
        effective_create_movie_arguments=dict(t_min=0, t_max=3.0, gray_screen_dur=0.5,
            cpd=0.04, temporal_f=2.0, theta=270.0, contrast=contrast_percent/100.0),
        checks={name: True for name in ('half_contrast_exact', 'combined_gray_exact', 'gray_is_zero',
            'retained_grating_exact', 'gray_join_frame_count_exact', 'first_nonzero_frame_follows_gray')})
    spike_config = deepcopy(config)
    spike_config['spike_generation'] = dict(trials=1, starting_noise_seed=12345)
    if include_rate_spike_generation:
        rate['config']['spike_generation'] = deepcopy(spike_config['spike_generation'])
    spikes = dict(status='SOURCE_SPIKES_EXPORTED_EXACT', source_units=17400, rate_shape=[17400, 3001],
        config=spike_config, starting_noise_seed=12345, rate_report_sha256='a'*64,
        rate_file_sha256='b'*64, prepared_runtime_sha256='c'*64,
        trials=[dict(trial=0, source_units=17400, inventory_roundtrip_exact=True,
                     event_order_exact=True, input_identity_sha256='f'*64,
                     event_comparison=dict(all_event_pairs_exact=True))])
    hashes = dict(rate_report_sha256='a'*64, rate_file_sha256='b'*64,
                  prepared_runtime_sha256='c'*64, spike_report_sha256='d'*64)
    return rate, movie, spikes, hashes


def command_report():
    rate, movie, spikes, hashes = source_reports()
    bound = validate_source_bindings(rate, movie, spikes, **hashes)
    return dict(bound, schema=SCHEMA, status='SOURCE_VISUAL_COMMANDS_EXACT_DEVELOPMENT',
        native_dt_ms=0.25, duration_ticks=12000, control_features=list(CONTROL_FEATURES),
        reconstruction=dict(all_values_exact=True, mismatch_count=0),
        movie_report_sha256='1'*64, source_movie_sha256='2'*64,
        control_semantics='Synthetic test of four-tick sample-and-hold; not a biological episode.')


def native_for(report):
    return {'generated_lgn': {key: report[key] for key in
            ('spike_report_sha256', 'input_identity_sha256', 'trial', 'config')}}


def write_controls_archive(root, report=None):
    directory = root/'controls'
    directory.mkdir()
    report = command_report() if report is None else report
    commands = make_commands(report['config']['stimulus'])
    commands.update(frame_mismatch_counts=np.zeros(3001, dtype=np.int64),
                    frame_max_absolute_differences=np.zeros(3001))
    np.savez_compressed(directory/'commands.npz', **commands)
    report['artifacts'] = {'commands.npz': sha256_file(directory/'commands.npz')}
    freeze_json(directory/'report.json', report)
    files = [dict(path='controls/'+name, bytes=(directory/name).stat().st_size,
                  sha256=sha256_file(directory/name)) for name in ('report.json', 'commands.npz')]
    freeze_json(root/'delivery.json', {'files': files})
    return report


class SourceCommandTests(unittest.TestCase):
    def test_gray_witness_requires_zero_movie_instead_of_grating_onset(self):
        rate, movie, spikes, hashes = source_reports(contrast_percent=0.0)
        del movie['checks']['first_nonzero_frame_follows_gray']
        movie['checks']['zero_contrast_movie_all_zero'] = True
        validate_source_bindings(rate, movie, spikes, **hashes)
        commands = make_commands(rate['config']['stimulus'])
        self.assertTrue(np.all(commands['control_values'][:, 0] == 0))
        self.assertTrue(np.all(reconstruct_frame(commands['control_values'][-1], [0, 1], [0, 1]) == 0))
        movie['checks']['zero_contrast_movie_all_zero'] = False
        with self.assertRaisesRegex(ValueError, 'MOVIE_CASE_OR_UNCHANGED_PREFIX_DIFFERS'):
            validate_source_bindings(rate, movie, spikes, **hashes)

    def test_onset_native_ticks_local_linspace_and_phase(self):
        commands = make_commands(stimulus())
        ticks, values = commands['control_ticks'], commands['control_values']
        self.assertTrue(np.array_equal(ticks, np.arange(3001, dtype=np.int64)*4))
        self.assertEqual(ticks[500], 2000)
        self.assertEqual(ticks[501], 2004)
        self.assertEqual(ticks[-1], 12000)
        self.assertTrue(np.all(values[:501, 0] == 0))
        self.assertTrue(np.all(values[501:, 0] == 0.8))
        self.assertTrue(np.all(values[:, 4] == 0))
        self.assertTrue(np.array_equal(values[501:, 5], np.linspace(0, 2.5, 2501)[1:]))
        self.assertEqual(values[-1, 5], 2.5)
        self.assertTrue(np.any(values[501:, 5] != (np.arange(3001)/1000.0)[501:]-0.5))

    def test_source_phase_and_theta_conventions(self):
        command = np.array([0.8, 0.04, 0.0, 180.0, 180.0, 0.0])
        # Source theta=180 projects on x; source phase=180 gives zero phase.
        frame = reconstruct_frame(command, np.array([0.0, 7.0]), np.array([0.0, 6.25]))
        self.assertTrue(np.array_equal(frame[:, 0], np.zeros(2)))
        self.assertTrue(np.array_equal(frame[:, 1], np.full(2, 0.8)))
        command[4] = 90.0
        self.assertEqual(reconstruct_frame(command, np.array([0.0]), np.array([0.0]))[0, 0], 0.8)

    def test_full_frame_comparison_preserves_mismatch_evidence(self):
        commands = make_commands(stimulus())
        commands.update(row_coordinates=np.array([0.0]), col_coordinates=np.array([0.0]))
        # A synthetic reference with exactly one altered frame, not source execution.
        data = np.stack([reconstruct_frame(row, commands['row_coordinates'], commands['col_coordinates'])
                         for row in commands['control_values']])
        exact = compare_source_movie(data, commands)
        self.assertTrue(exact['all_values_exact'])
        data[501, 0, 0] += 0.125
        changed = compare_source_movie(data, commands)
        self.assertFalse(changed['all_values_exact'])
        self.assertEqual(changed['mismatch_count'], 1)
        self.assertEqual(changed['affected_frame_indices'], [501])
        self.assertEqual(commands['frame_mismatch_counts'][501], 1)

    def test_source_hashes_inventory_and_configuration_mismatches_reject(self):
        rate, movie, spikes, hashes = source_reports()
        bound = validate_source_bindings(rate, movie, spikes, **hashes)
        self.assertNotEqual(bound['movie_prepared_runtime_sha256'], bound['prepared_runtime_sha256'])
        for key in ('rate_report_sha256', 'rate_file_sha256', 'prepared_runtime_sha256'):
            changed = deepcopy(spikes)
            changed[key] = '9'*64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'SOURCE_BINDING_DIFFERS'):
                validate_source_bindings(rate, movie, changed, **hashes)
        changed = deepcopy(spikes)
        changed['config']['stimulus']['contrast_percent'] = 40.0
        with self.assertRaisesRegex(ValueError, 'BOUND_SPIKE_CASE_REQUIRED'):
            validate_source_bindings(rate, movie, changed, **hashes)
        changed = deepcopy(rate)
        changed['source_unit_ids'][1] = 0
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SOURCE_RATE_CASE_REQUIRED'):
            validate_source_bindings(changed, movie, spikes, **hashes)
        changed = deepcopy(movie)
        changed['effective_create_movie_arguments']['theta'] = 90.0
        with self.assertRaisesRegex(ValueError, 'MOVIE_CASE_OR_UNCHANGED_PREFIX'):
            validate_source_bindings(rate, changed, spikes, **hashes)

    def test_full_rate_config_with_sampling_metadata_binds_40_percent_case(self):
        rate, movie, spikes, hashes = source_reports(
            contrast_percent=40.0, include_rate_spike_generation=True)
        original = deepcopy((rate, movie, spikes))
        bound = validate_source_bindings(rate, movie, spikes, **hashes)
        self.assertEqual(bound['config'], spikes['config'])
        self.assertEqual(bound['config']['spike_generation'],
                         {'trials': 1, 'starting_noise_seed': 12345})
        self.assertEqual(bound['stimulus']['contrast_percent'], 40.0)
        self.assertEqual(bound['trial'], 0)
        self.assertEqual(bound['input_identity_sha256'], spikes['trials'][0]['input_identity_sha256'])
        self.assertEqual((rate, movie, spikes), original)
        commands = make_commands(bound['stimulus'])
        self.assertTrue(np.all(commands['control_values'][:501, 0] == 0))
        self.assertTrue(np.all(commands['control_values'][501:, 0] == 0.4))

    def test_full_rate_config_rejects_sampling_and_rate_setting_mismatches(self):
        rate, movie, spikes, hashes = source_reports(
            contrast_percent=40.0, include_rate_spike_generation=True)
        for change in ('rate_seed', 'rate_trials', 'rate_sampling_extra',
                       'spike_seed', 'spike_contrast', 'spike_converter'):
            changed_rate, changed_spikes = deepcopy(rate), deepcopy(spikes)
            if change == 'rate_seed':
                changed_rate['config']['spike_generation']['starting_noise_seed'] += 1
            elif change == 'rate_trials':
                changed_rate['config']['spike_generation']['trials'] += 1
            elif change == 'rate_sampling_extra':
                changed_rate['config']['spike_generation']['extra_source_setting'] = True
            elif change == 'spike_seed':
                changed_spikes['config']['spike_generation']['starting_noise_seed'] += 1
                changed_spikes['starting_noise_seed'] += 1
            elif change == 'spike_contrast':
                changed_spikes['config']['stimulus']['contrast_percent'] = 80.0
            else:
                changed_spikes['config']['converter_python'] = '/different/converter/python3'
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'BOUND_SPIKE_CASE_REQUIRED'):
                validate_source_bindings(changed_rate, movie, changed_spikes, **hashes)

    def test_native_binding_rejects_historical_wrong_case_and_nonpass(self):
        report = command_report()
        validate_native_binding(report, native_for(report))
        with self.assertRaisesRegex(ValueError, 'REQUIRE_GENERATED_INPUT'):
            validate_native_binding(report, {})
        for key, value in (('spike_report_sha256', '9'*64),
                           ('input_identity_sha256', '9'*64), ('trial', 1)):
            native = native_for(report)
            native['generated_lgn'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'NATIVE_GENERATED_INPUT_DIFFERS'):
                validate_native_binding(report, native)
        native = deepcopy(native_for(report))
        native['generated_lgn']['config']['spike_generation']['starting_noise_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'NATIVE_GENERATED_INPUT_DIFFERS'):
            validate_native_binding(report, native)
        failed = deepcopy(report)
        failed['reconstruction']['all_values_exact'] = False
        with self.assertRaisesRegex(ValueError, 'EXACT_DEVELOPMENT_COMMANDS_REQUIRED'):
            validate_native_binding(failed, native_for(report))

    def test_receipt_loader_binds_controls_to_native_generated_input(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-v1-controls-') as temporary:
            root = Path(temporary)
            report = write_controls_archive(root)
            loaded = load_visual_controls(root, native_for(report))
            self.assertEqual(loaded['control_values'].shape, (3001, 6))
            self.assertEqual(loaded['source_control_record']['input_identity_sha256'], report['input_identity_sha256'])
            self.assertNotIn('episode_id', loaded['source_control_record'])
            with self.assertRaisesRegex(ValueError, 'REQUIRE_GENERATED_INPUT'):
                load_visual_controls(root, {})


class SyntheticCircuitWindowTests(unittest.TestCase):
    def test_three_unit_windows_onset_endpoint_and_privileged_separation(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-v1-control-window-') as temporary:
            root = Path(temporary)
            controls_root = root/'control_receipt'
            controls_root.mkdir()
            report = write_controls_archive(controls_root)
            controls = load_visual_controls(controls_root, native_for(report))
            provenance = {'connectivity_identity': 'synthetic-three-unit-graph',
                          'parameter_identity': 'synthetic-test-parameters'}
            units = CircuitUnits('synthetic_circuit', np.arange(3), np.array(['toy', 'toy', 'toy']),
                np.zeros(3, dtype=np.int64), ('toy',), 'Synthetic test positions',
                'synthetic_revision', circuit_provenance=provenance)
            episode = CircuitEpisode('synthetic_episode', 'synthetic_circuit', 'synthetic_block',
                'development', 12000, 0.25,
                np.array([0, 2003, 2004, 2005, 12000], dtype=np.int64),
                np.array([0, 1, 2, 2, 0], dtype=np.int64),
                controls['control_ticks'], controls['control_values'], controls['control_features'],
                source_record=dict(complete_unit_recording=True, simulation_completed=True,
                    recorded_until_tick=12000, expected_duration_ticks=12000, core_unit_count=3,
                    source_revision='synthetic_revision', circuit_provenance=provenance,
                    synthetic_fixture=True),
                privileged={'incoming_raw_event_time_ms': np.array([0.001, 2.0]),
                            'incoming_source_ids': np.array([0, 1]),
                            'final_endpoint_ticks': np.array([12000])})
            dataset_root = root/'dataset'
            binding = dict(schema=CIRCUIT_SCHEMA, scope='source_reproduction',
                source_revision='synthetic_revision', sample_unit='complete_mechanistic_episode',
                synthetic_fixture=True)
            with CircuitArchiveWriter(dataset_root, binding) as writer:
                writer.register_circuit(units)
                writer.add(episode)
            dataset = CircuitDataset(dataset_root, splits=('development',))
            at_onset = dataset.window('synthetic_episode', boundary_tick=2004,
                                      history_ticks=4, horizon_ticks=4, bin_ticks=1)
            expected = np.vstack([np.tile(controls['control_values'][500], (4, 1)),
                                  np.tile(controls['control_values'][501], (4, 1))])
            self.assertTrue(np.array_equal(at_onset['known_controls'], expected))
            self.assertEqual(at_onset['history'].sum(), 1)
            self.assertEqual(at_onset['targets'].sum(), 2)
            crossing = dataset.window('synthetic_episode', boundary_tick=2003,
                                      history_ticks=3, horizon_ticks=3, bin_ticks=3)
            expected = np.vstack([controls['control_values'][500],
                controls['control_values'][500]*(1/3)+controls['control_values'][501]*(2/3)])
            self.assertTrue(np.array_equal(crossing['known_controls'], expected))
            ending = dataset.window('synthetic_episode', boundary_tick=11996,
                                    history_ticks=4, horizon_ticks=4, bin_ticks=1)
            self.assertEqual(ending['targets'].sum(), 0)
            self.assertTrue(np.array_equal(ending['known_controls'][-1], controls['control_values'][2999]))
            for window in (at_onset, crossing, ending):
                self.assertNotIn('privileged', window)
                self.assertNotIn('incoming_raw_event_time_ms', window)
                self.assertNotIn('incoming_source_ids', window)
            row = dataset.episodes['synthetic_episode']
            shard = dataset.shards[row['shard']]
            with h5py.File(dataset_root/shard['path'], 'r') as source:
                saved = source['synthetic_episode']
                self.assertEqual(saved['control_ticks'][-1], 12000)
                self.assertEqual(saved['control_values'][-1, 5], 2.5)
                self.assertEqual(saved['spike_ticks'][-1], 12000)
                self.assertEqual(saved['privileged/final_endpoint_ticks'][0], 12000)



def constructed_reports():
    rate, movie, lgn, hashes = source_reports(include_rate_spike_generation=True)
    rate['config']['stimulus']['direction_deg'] = 315.0
    rate['config']['background'] = dict(kind='source_constant_rate_sampler', rate_hz=10.0,
        rate_clock_step_ms=0.25, starting_noise_seed=24680)
    lgn['config'] = deepcopy(rate['config'])
    movie['effective_create_movie_arguments']['theta'] = 315.0
    movie['rate_report_sha256'] = '8'*64
    construction = dict(delivery_sha256='9'*64, mode='full',
        files={'primary/glif/bkg_nodes.h5':dict(bytes=123, sha256='a'*64)})
    rate['input_construction_binding'] = deepcopy(construction)
    lgn['input_construction_binding'] = deepcopy(construction)
    lgn['config_sha256'] = '0'*64
    lgn['trials'][0].update(plan='trial_0000.npz',sonata='trial_0000.h5')
    bundle = dict(status=visual_controls.CONSTRUCTED_COMPLETE, exit_code=0,
        complete_lgn_population=True, config=deepcopy(lgn['config']), config_sha256='0'*64,
        input_construction_binding=deepcopy(construction),
        source_clock=dict(duration_ms=3000.,resolution_ms=.25),
        rate_report_sha256=hashes['rate_report_sha256'],
        prepared_runtime_sha256=hashes['prepared_runtime_sha256'],
        lgn_report_sha256=hashes['spike_report_sha256'],
        artifacts={'lgn/report.json':hashes['spike_report_sha256']},
        lgn_noise_seed=12345,background_noise_seed=24680,
        populations=dict(lgn=dict(lgn['trials'][0],plan='lgn/trial_0000.npz',sonata='lgn/trial_0000.h5'),
            bkg=dict(source_units=1,input_identity_sha256='5'*64,
                     plan='bkg/trial_0000.npz',sonata='bkg/trial_0000.h5')),
        bundle_report_sha256='6'*64,input_identity_sha256='7'*64)
    return rate,movie,lgn,bundle,hashes


def constructed_command_report():
    rate,movie,lgn,bundle,hashes=constructed_reports()
    bound=validate_constructed_source_bindings(rate,movie,lgn,bundle,**hashes)
    relatives=['visual_inputs/report.json'] + ['visual_inputs/'+bundle['populations'][name][key]
        for name in ('lgn','bkg') for key in ('plan','sonata')]
    return dict(bound,schema=SCHEMA,status='SOURCE_VISUAL_COMMANDS_EXACT_DEVELOPMENT',
        native_dt_ms=.25,duration_ticks=12000,control_features=list(CONTROL_FEATURES),
        source_command_representation_verified=True,scientific_qualification=False,
        source_fidelity_qualified=False,visual_action_mapping_qualified=False,production_domain_qualified=False,
        reconstruction=dict(all_values_exact=True,mismatch_count=0),
        movie_report_sha256='1'*64,source_movie_sha256='2'*64,movie_witness_artifact_sha256='3'*64,
        constructed_input_delivery_sha256='4'*64,
        constructed_input_files={name:dict(bytes=100+i,sha256=str(i)*64) for i,name in enumerate(relatives)},
        control_semantics='Synthetic 315 degree development commands; no biological evidence.')


def pointnet_for(report):
    binding={key:deepcopy(report[key]) for key in ('bundle_report_sha256','bundle_status',
        'input_construction_binding','config','source_clock','complete_lgn_population')}
    binding.update(delivery_sha256=report['constructed_input_delivery_sha256'],
        files=deepcopy(report['constructed_input_files']),input_identity_sha256=report['merged_input_identity_sha256'])
    return dict(input_mode='constructed_visual_inputs',input_event_schema='packed_source_events_v1',
                bindings={'spike_inputs':binding})


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def _deliver(root,relatives):
    _write_json(root/'delivery.json',dict(exit_code=0,files=[dict(path=name,
        bytes=(root/name).stat().st_size,sha256=sha256_file(root/name)) for name in relatives]))


def full_constructed_archive(root):
    """Full inventories with two synthetic events and a sparse, unwritten rate matrix."""
    rate,movie,lgn,bundle,hashes=constructed_reports()
    rate_root=root/'rates';bundle_root=root/'bundle'
    runtime=rate_root/'prepared'
    _write_json(runtime/'prepared.json',dict(status='SOURCE_RUNTIME_PREPARED',files=[]))
    (runtime/'lgnmodel').mkdir();(runtime/'lgnmodel/movie.py').write_text('# synthetic runtime fixture\n')
    rate_file=rate_root/'probe/firing_rates_f_tot.h5';rate_file.parent.mkdir()
    with h5py.File(rate_file,'w') as f:
        f['time']=np.arange(FRAME_COUNT)/1000.
        f.create_dataset('firing_rates_Hz',shape=(17400,FRAME_COUNT),dtype='f8',chunks=(1,FRAME_COUNT))
    _write_json(rate_root/'probe/probe.json',rate)
    hashes.update(rate_report_sha256=sha256_file(rate_root/'probe/probe.json'),
        rate_file_sha256=sha256_file(rate_file),prepared_runtime_sha256=sha256_file(runtime/'prepared.json'))
    rows={}
    for name,n in [('lgn',17400),('bkg',1)]:
        folder=bundle_root/'visual_inputs'/name;folder.mkdir(parents=True)
        ids=np.arange(n,dtype=np.int64)
        trains=[np.array([.125 if name=='lgn' else .5])]+[np.empty(0) for _ in range(n-1)]
        seeds=[dict(source_id=int(i),primary_seed=1,poisson_seed=None,sampler='bernoulli') for i in ids]
        rows[name]=write_spike_trial(ids,trains,seeds,3000.,folder/'trial_0000',population_name=name)
    lgn.update({key:hashes[key] for key in ('rate_report_sha256','rate_file_sha256','prepared_runtime_sha256')})
    lgn['trials']=[dict(rows['lgn'],trial=0)]
    _write_json(bundle_root/'visual_inputs/lgn/report.json',lgn)
    hashes['spike_report_sha256']=sha256_file(bundle_root/'visual_inputs/lgn/report.json')
    bundle.update(rate_report_sha256=hashes['rate_report_sha256'],
        prepared_runtime_sha256=hashes['prepared_runtime_sha256'],lgn_report_sha256=hashes['spike_report_sha256'])
    bundle['populations']={name:dict(lgn['trials'][0] if name=='lgn' else rows[name],
        plan=name+'/'+rows[name]['plan'],sonata=name+'/'+rows[name]['sonata']) for name in ('lgn','bkg')}
    relatives=['visual_inputs/lgn/report.json']+['visual_inputs/'+bundle['populations'][name][key]
        for name in ('lgn','bkg') for key in ('plan','sonata')]
    bundle['artifacts']={str(Path(name).relative_to('visual_inputs')):sha256_file(bundle_root/name) for name in relatives}
    # These identities are computed by load_inputs, not stored in the source bundle report.
    bundle.pop('bundle_report_sha256');bundle.pop('input_identity_sha256')
    _write_json(bundle_root/'visual_inputs/report.json',bundle)
    _deliver(bundle_root,['visual_inputs/report.json',*relatives])
    _deliver(rate_root,['probe/probe.json','probe/firing_rates_f_tot.h5',
                        'prepared/prepared.json','prepared/lgnmodel/movie.py'])
    return rate_root,bundle_root,rate,movie,lgn,bundle


class ConstructedCommandTests(unittest.TestCase):
    def test_lgn_background_and_merged_identities_remain_distinct(self):
        rate,movie,lgn,bundle,hashes=constructed_reports()
        report=validate_constructed_source_bindings(rate,movie,lgn,bundle,**hashes)
        self.assertEqual(report['lgn_input_identity_sha256'],'f'*64)
        self.assertEqual(report['background_input_identity_sha256'],'5'*64)
        self.assertEqual(report['merged_input_identity_sha256'],'7'*64)
        self.assertEqual(report['lgn_report_sha256'],'d'*64)
        self.assertEqual(report['bundle_report_sha256'],'6'*64)
        self.assertNotIn('input_identity_sha256',report)
        self.assertNotIn('spike_report_sha256',report)
        self.assertEqual(report['source_family'],CONSTRUCTED_FAMILY)

    def test_constructed_noise_blocks_retain_rate_and_movie_bindings(self):
        for lgn_seed, bkg_seed in [(22345, 24680), (12345, 22346), (22345, 22346)]:
            rate, movie, lgn, bundle, hashes = constructed_reports()
            original_rate = deepcopy(rate)
            lgn['config']['spike_generation']['starting_noise_seed'] = lgn_seed
            lgn['config']['background']['starting_noise_seed'] = bkg_seed
            lgn['starting_noise_seed'] = lgn_seed
            bundle['config'] = deepcopy(lgn['config'])
            bundle.update(lgn_noise_seed=lgn_seed, background_noise_seed=bkg_seed)
            with self.subTest(lgn_seed=lgn_seed, bkg_seed=bkg_seed):
                bound = validate_constructed_source_bindings(rate, movie, lgn, bundle, **hashes)
                self.assertEqual(rate, original_rate)
                self.assertEqual(bound['rate_config'], rate['config'])
                self.assertEqual(bound['config'], lgn['config'])
                self.assertEqual(bound['stimulus'], rate['config']['stimulus'])
                self.assertEqual(bound['input_construction_binding'], rate['input_construction_binding'])
                with self.assertRaisesRegex(ValueError, 'BOUND_SPIKE_CASE_REQUIRED'):
                    validate_source_bindings(rate, movie, lgn, **hashes)

    def test_constructed_noise_reuse_preserves_complete_configuration_and_seed_types(self):
        for target in ('rate', 'lgn'):
            for section in ('spike_generation', 'background'):
                for value in (None, True, 12345., '12345', 'absent'):
                    rate, movie, lgn, bundle, hashes = constructed_reports()
                    config = rate['config'] if target == 'rate' else lgn['config']
                    if value == 'absent': del config[section]['starting_noise_seed']
                    else: config[section]['starting_noise_seed'] = value
                    with self.subTest(target=target, section=section, value=value), self.assertRaises(ValueError):
                        validate_constructed_source_bindings(rate, movie, lgn, bundle, **hashes)
        for path, value in [(('background', 'rate_clock_step_ms'), .5),
                            (('background', 'rate_hz'), 1000.),
                            (('spike_generation', 'trials'), 2),
                            (('unit_coverage',), 'model_representatives')]:
            rate, movie, lgn, bundle, hashes = constructed_reports()
            destination = lgn['config']
            for key in path[:-1]: destination = destination[key]
            destination[path[-1]] = value
            bundle['config'] = deepcopy(lgn['config'])
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_constructed_source_bindings(rate, movie, lgn, bundle, **hashes)

    def test_constructed_rate_spike_bundle_and_movie_mismatches_reject(self):
        for change in ('construction','bkg_seed','bkg_config','config_hash','lgn_identity',
                       'lgn_report','rate_report','runtime','clock','representatives','old_movie'):
            rate,movie,lgn,bundle,hashes=constructed_reports()
            if change=='construction':bundle['input_construction_binding']['delivery_sha256']='1'*64
            elif change=='bkg_seed':bundle['background_noise_seed']+=1
            elif change=='bkg_config':bundle['config']['background']['rate_hz']+=1
            elif change=='config_hash':bundle['config_sha256']='1'*64
            elif change=='lgn_identity':bundle['populations']['lgn']['input_identity_sha256']='1'*64
            elif change=='lgn_report':bundle['lgn_report_sha256']='1'*64
            elif change=='rate_report':bundle['rate_report_sha256']='1'*64
            elif change=='runtime':bundle['prepared_runtime_sha256']='1'*64
            elif change=='clock':bundle['source_clock']['resolution_ms']=1.
            elif change=='representatives':lgn['source_units']=14
            else:movie['effective_create_movie_arguments']['theta']=270.
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_constructed_source_bindings(rate,movie,lgn,bundle,**hashes)

    def test_pointnet_native_binding_rejects_wrong_construction_background_and_merged_case(self):
        report=constructed_command_report()
        validate_native_binding(report,pointnet_for(report),native_family='pointnet')
        for change in ('construction','bkg_file','merged_identity','bundle_report','config','diagnostic'):
            native=pointnet_for(report);binding=native['bindings']['spike_inputs']
            if change=='construction':binding['input_construction_binding']['delivery_sha256']='1'*64
            elif change=='bkg_file':binding['files']['visual_inputs/bkg/trial_0000.npz']['sha256']='9'*64
            elif change=='merged_identity':binding['input_identity_sha256']=report['lgn_input_identity_sha256']
            elif change=='bundle_report':binding['bundle_report_sha256']='1'*64
            elif change=='config':binding['config']['background']['rate_hz']+=1
            else:native['input_mode']='diagnostic_pulses'
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_native_binding(report,native,native_family='pointnet')
        with self.assertRaises(ValueError):
            validate_native_binding(command_report(),pointnet_for(report),native_family='pointnet')
        with self.assertRaises(ValueError):
            validate_native_binding(report,native_for(command_report()))

    def test_pointnet_receipt_loader_preserves_representation_only_qualification(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-constructed-controls-') as temporary:
            root=Path(temporary);report=write_controls_archive(root,constructed_command_report())
            loaded=load_visual_controls(root,pointnet_for(report),native_family='pointnet')
            record=loaded['source_control_record'];values=loaded['control_values']
            self.assertTrue(record['source_command_representation_verified'])
            for key in ('scientific_qualification','source_fidelity_qualified','visual_action_mapping_qualified','production_domain_qualified'):
                self.assertIs(record[key],False)
            self.assertTrue(np.all(values[:,3]==315.))
            self.assertTrue(np.all(values[:501,0]==0.))
            self.assertEqual(loaded['control_ticks'][501],2004)
            self.assertEqual(values[501,5],.001)
            self.assertEqual(record['background_input_identity_sha256'],'5'*64)

    def test_full_constructed_export_uses_actual_bundle_and_exact_movie_clocks(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-full-visual-join-') as temporary:
            root=Path(temporary)
            rate_root,bundle_root,rate,movie_report,_,_=full_constructed_archive(root)
            movie_root=root/'movie';(movie_root/'movie').mkdir(parents=True)
            commands=make_commands(rate['config']['stimulus'])
            rows=np.arange(48,dtype=float);cols=np.arange(96,dtype=float)
            data=np.stack([reconstruct_frame(value,rows,cols) for value in commands['control_values']])
            movie=SimpleNamespace(data=data,t_range=np.arange(FRAME_COUNT)*.001,
                row_range=rows,col_range=cols,frame_rate=1000.,labels=['time','row','col'],units=['s','pixel','pixel'])
            indices=np.array([0,500,501,3000])
            witness=movie_root/'movie/movie_witness.npz'
            np.savez_compressed(witness,frame_indices=indices,selected_frames=data[indices],
                movie_clock_s=movie.t_range,rate_clock_s=np.arange(FRAME_COUNT)/1000.,
                row_coordinates=rows,col_coordinates=cols,frame_min=data.min(axis=(1,2)),frame_max=data.max(axis=(1,2)),
                origin_pixel=data[:,0,0],center_pixel=data[:,24,48])
            movie_report['artifacts']={'movie_witness.npz':sha256_file(witness)}
            _write_json(movie_root/'movie/report.json',movie_report)
            _deliver(movie_root,['movie/report.json','movie/movie_witness.npz'])
            class SyntheticGrating:
                def __init__(self,*args):pass
                def create_movie(self,**arguments):
                    if arguments!=movie_report['effective_create_movie_arguments']:
                        raise AssertionError('wrong stimulus arguments')
                    return movie
            functions=SimpleNamespace(GratingMovie=SyntheticGrating)
            controls_root=root/'export';controls_root.mkdir()
            with patch.object(visual_controls,'load_prepared_runtime',return_value=(functions,None)), \
                    patch('population_dynamics.v1_circuit_stimulus.load_inputs',wraps=load_inputs) as consumer:
                report=visual_controls.export_constructed_visual_controls(rate_root,movie_root,bundle_root,controls_root/'controls')
            self.assertIs(consumer.call_args.kwargs['require_complete'],True)
            self.assertEqual(report['status'],'SOURCE_VISUAL_COMMANDS_EXACT_DEVELOPMENT')
            self.assertEqual(report['reconstruction']['compared_values'],FRAME_COUNT*48*96)
            self.assertEqual(report['reconstruction']['mismatch_count'],0)
            self.assertFalse(report['movie_rate_clock_exact'])
            self.assertGreater(report['movie_rate_clock_mismatches'],0)
            self.assertEqual(report['gray_frames'],501)
            self.assertEqual(report['first_grating_tick'],2004)
            self.assertEqual(report['first_grating_local_time_s'],.001)
            merged,bundle=load_inputs(bundle_root,rate['input_construction_binding'])
            self.assertEqual(len(merged['source_id']),17401)
            self.assertEqual(np.count_nonzero(merged['source_counts']==0),17399)
            self.assertEqual(report['merged_input_identity_sha256'],bundle['input_identity_sha256'])
            self.assertEqual(report['lgn_input_identity_sha256'],bundle['populations']['lgn']['input_identity_sha256'])
            self.assertEqual(report['background_input_identity_sha256'],bundle['populations']['bkg']['input_identity_sha256'])
            self.assertNotEqual(report['lgn_input_identity_sha256'],report['merged_input_identity_sha256'])
            for key in ('scientific_qualification','source_fidelity_qualified','visual_action_mapping_qualified','production_domain_qualified'):
                self.assertIs(report[key],False)
            self.assertEqual(report['production_episodes'],0)
            _deliver(controls_root,['controls/report.json','controls/commands.npz'])
            loaded=load_visual_controls(controls_root,pointnet_for(report),native_family='pointnet')
            self.assertEqual(loaded['control_values'].shape,(3001,6))
            # An internally resealed wrong BKG identity must still fail the reused bundle consumer.
            raw=json.loads((bundle_root/'visual_inputs/report.json').read_text())
            raw['populations']['bkg']['input_identity_sha256']='e'*64
            _write_json(bundle_root/'visual_inputs/report.json',raw)
            delivery=json.loads((bundle_root/'delivery.json').read_text())
            _deliver(bundle_root,[row['path'] for row in delivery['files']])
            with self.assertRaisesRegex(ValueError,'INVENTORY_OR_IDENTITY'):
                load_inputs(bundle_root,rate['input_construction_binding'],require_complete=True)
            raw['status']='CONSTRUCTED_VISUAL_REPRESENTATIVES_COMPLETE_DEVELOPMENT'
            raw['complete_lgn_population']=False
            _write_json(bundle_root/'visual_inputs/report.json',raw)
            _deliver(bundle_root,[row['path'] for row in delivery['files']])
            with self.assertRaisesRegex(ValueError,'COMPLETE_VISUAL_POPULATION'):
                load_inputs(bundle_root,rate['input_construction_binding'],require_complete=True)

    def test_visual_cli_routes_constructed_and_legacy_sources(self):
        common=['--rate-archive','rates','--movie-archive','movie','--output','controls']
        for flag,name in [('--spike-archive','export_visual_controls'),
                          ('--constructed-input-archive','export_constructed_visual_controls')]:
            with self.subTest(flag=flag),patch.object(visual_controls,name,return_value={'exit_code':0}) as exporter:
                self.assertEqual(visual_controls.main(common+[flag,'inputs']),0)
                self.assertEqual(exporter.call_args.args,(Path('rates'),Path('movie'),Path('inputs'),Path('controls')))
        with self.assertRaises(SystemExit):
            visual_controls.main(common+['--spike-archive','spikes','--constructed-input-archive','bundle'])


if __name__ == '__main__':
    unittest.main()
