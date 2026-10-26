"""Three cortical/two LGN unit fixtures; no native execution or scientific samples."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import v1_cortical_control_pair as cortical
from .provenance import freeze_json, sha256_file


def configuration(contrast):
    return dict(scope='development_source_runtime_qualification', unit_coverage='all_source_units',
        stimulus=dict(duration_s=3.0, gray_screen_s=0.5, contrast_percent=contrast,
                      spatial_frequency_cycles_per_degree=0.04, temporal_frequency_hz=2.0, direction_deg=270.0),
        spike_generation=dict(starting_noise_seed=12345, trials=1))


def write_delivery(root, relatives):
    freeze_json(root/'delivery.json', dict(status='PASS', exit_code=0, synthetic_fixture=True,
        files=[dict(path=name, bytes=(root/name).stat().st_size, sha256=sha256_file(root/name)) for name in relatives]))


def fixtures(root, variant=None):
    pair = dict(status='SOURCE_SPIKE_CONTROL_PAIR_EXACT_DEVELOPMENT', exit_code=0, synthetic_fixture=True,
        baseline_config=configuration(80.0), control_config=configuration(40.0),
        source_array_references={role: dict(spike_report_sha256=report_hash*64,
            input_identity_sha256=input_hash*64, trial=0) for role, report_hash, input_hash in
            (('baseline', 'a', 'b'), ('control', 'c', 'd'))},
        comparison=dict(source_units=2, source_unit_ids=[0, 1], pair_exact=True,
            recorded_sampler_initialization=dict(exact=True),
            prefix=dict(exact=True, boundary_ms=501.0, global_comparison=dict(all_event_pairs_exact=True)),
            per_unit=[dict(source_id=i, initialization_exact=True, prefix=dict(event_multiset_exact=True)) for i in range(2)]))
    if variant == 'incomplete_pair':
        pair['comparison']['per_unit'].pop()
    input_root = root/'input_pair'
    (input_root/'spike_pair').mkdir(parents=True)
    freeze_json(input_root/'spike_pair/report.json', pair)
    write_delivery(input_root, ['spike_pair/report.json'])
    archives = []
    for role in ('baseline', 'control'):
        control = role == 'control'
        archive = root/role
        (archive/'network').mkdir(parents=True)
        for relative in cortical.NATIVE_SOURCE_FILES:
            source = archive/relative
            source.parent.mkdir(parents=True, exist_ok=True)
            contents = '# Synthetic source fixture: ' + relative + '\n'
            if control and variant == 'helper_code' and source.name == 'v1_nest_time_semantics.py':
                contents += '# Changed helper implementation.\n'
            source.write_bytes(contents.encode('utf-8'))
        identity = dict(source_id=np.arange(3, dtype=np.int64), native_id=np.array([11, 12, 13]),
            node_type_id=np.array([100, 100, 101]), initial_vm_mV=np.array([-70., -70., -70.]))
        if control and variant == 'initial_vm':
            identity['initial_vm_mV'][1] += 0.125
        if control and variant == 'identity':
            identity['source_id'][1] = 0
        ids = np.array([0, 0, 1, 0, 1], dtype=np.int64)
        times = np.array([10., 500.75, 501., 1100. if control else 1000., 3000.])
        if control and variant == 'prefix_time':
            times[1] = 500.5
        if control and variant == 'prefix_count':
            ids, times = np.delete(ids, 1), np.delete(times, 1)
        activity = dict(spike_source_id=ids, spike_time_ms=times,
            spike_counts=np.bincount(ids, minlength=3),
            final_vm_mV=np.array([-61., -64., -70.]) if control else np.array([-60., -65., -70.]))
        lgn0 = np.array([1., 1., 501., 900. if control else 800.])
        lgn1 = np.array([2., 700.])
        background = np.array([4., 500., 1000., 3000.])
        if control and variant == 'lgn_prefix':
            lgn0[0] = 1.25
        if control and variant == 'background':
            background[2] = 1001.
        counts = np.array([len(lgn0), len(lgn1), len(background)], dtype=np.int64)
        plan = dict(population=np.array([0, 0, 1]), source_id=np.array([0, 1, 0]), source_counts=counts,
            source_event_unit=np.repeat(np.arange(3), counts), source_event_time_ms=np.concatenate((lgn0, lgn1, background)))
        input_identity = dict(population=plan['population'], source_id=plan['source_id'],
            native_id=np.array([101, 102, 103]), source_event_counts=counts)
        arrays = {'cortical_identity.npz': identity, 'cortical_activity.npz': activity,
                  'input_identity.npz': input_identity, 'source_inputs.npz': plan}
        for name, values in arrays.items():
            np.savez_compressed(archive/'network'/name, **values)
        generated = dict(pair['source_array_references'][role],
            config=configuration(40.0 if control else 80.0), generated_units=2,
            background_events_unchanged=True, input_edge_objects_unchanged=True,
            original_plan_sha256='e'*64, input_origin_sha256='f'*64)
        if control and variant == 'input_binding':
            generated['input_identity_sha256'] = '9'*64
        native = dict(status='NATIVE_SOURCE_GRAPH_DIAGNOSTIC_COMPLETE', full_cortical_network_executed=True,
            implementation_sha256='1'*64 if variant == 'implementation_binding' else sha256_file(archive/cortical.NATIVE_SOURCE_FILES[0]),
            nest_version='3.10.0', threads=4, connection_batch_size=100000,
            kernel_rng_seed=1235 if control and variant == 'rng' else 1234,
            coupled_cortical_neurons=3, native_duration_ms=3000.0, input_generators=3,
            input_events=int(counts.sum()), recurrent_connections={'source_synapses': 3},
            input_connections=[{'population_index': 0}, {'population_index': 1}],
            native_total_connections=12, observation_scope='Synthetic spikes and boundary Vm',
            kernel_delays_ms={'min_delay': 0.25, 'max_delay': 3.75},
            parameter_initialization_source='Synthetic fixed constructor parameters', generated_lgn=generated,
            artifacts={name: sha256_file(archive/'network'/name) for name in arrays})
        freeze_json(archive/'network/report.json', native)
        for relative in cortical.SOURCE_RECEIPTS:
            value = {'synthetic_fixture': True, 'source': relative}
            if relative == 'inputs/input_identity.json':
                value['source_config_clock'] = {'duration': 3000.0, 'dt': 0.25}
            if control and variant == 'graph' and relative == 'graph/graph_identity.json':
                value['changed_source'] = True
            freeze_json(archive/relative, value)
        source_rows = list(cortical.NATIVE_SOURCE_FILES)
        if control and variant == 'source_missing':
            source_rows.remove('Libs/population_dynamics/v1_nest_time_semantics.py')
        write_delivery(archive, ['network/report.json', *('network/'+name for name in arrays),
                                 *cortical.SOURCE_RECEIPTS, *source_rows])
        archives.append(archive)
    return *archives, input_root


class CorticalPairTests(unittest.TestCase):
    def execute(self, root, variant=None):
        first, second, input_pair = fixtures(root, variant)
        before = [sha256_file(path/'network/cortical_activity.npz') for path in (first, second)]
        with mock.patch.multiple(cortical, CORTICAL_UNITS=3, LGN_UNITS=2):
            report = cortical.compare_archives(first, second, input_pair, root/'output')
        self.assertEqual(before, [sha256_file(path/'network/cortical_activity.npz') for path in (first, second)])
        self.assertEqual(json.loads((root/'output/report.json').read_text())['exit_code'], report['exit_code'])
        with np.load(root/'output/per_unit.npz', allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        return report, arrays

    def test_exact_prefix_with_later_response_silence_and_duplicate_inputs(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cortical-pair-') as temporary:
            report, arrays = self.execute(Path(temporary))
            expected_sources = {relative: sha256_file(Path(temporary)/'baseline'/relative)
                                for relative in cortical.NATIVE_SOURCE_FILES}
        self.assertEqual(report['matched_native_source_sha256'], expected_sources)
        self.assertEqual(report['exit_code'], 0)
        self.assertEqual(report['status'], 'CORTICAL_CONTROL_PAIR_EXACT_DEVELOPMENT')
        self.assertEqual(report['prefix_end_tick'], 2004)
        self.assertTrue(report['comparison']['prefix']['event_multisets_exact'])
        self.assertEqual(report['comparison']['later']['differing_units'], 1)
        self.assertFalse(report['comparison']['terminal_vm']['exact'])
        self.assertTrue(np.array_equal(arrays['source_id'], [0, 1, 2]))
        self.assertEqual(arrays['prefix_baseline_count'][2], 0)
        self.assertEqual(arrays['later_control_count'][2], 0)
        self.assertEqual(report['actual_input_comparisons']['lgn_prefix']['expected_events'], 3)
        self.assertTrue(report['actual_input_comparisons']['background']['all_event_pairs_exact'])
        self.assertFalse(report['intermediate_voltage_recorded'])
        self.assertFalse(report['hidden_random_draws_equality_claimed'])

    def test_prefix_timestamp_and_count_differences_preserve_complete_arrays(self):
        for variant in ('prefix_time', 'prefix_count'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(prefix='synthetic-cortical-prefix-') as temporary:
                report, arrays = self.execute(Path(temporary), variant)
            self.assertEqual(report['status'], 'CORTICAL_CONTROL_PAIR_DIFFERS_DIAGNOSTIC')
            self.assertEqual(report['exit_code'], 2)
            self.assertEqual(len(arrays['source_id']), 3)
            self.assertFalse(arrays['prefix_event_multiset_exact'][0])
            if variant == 'prefix_count':
                self.assertFalse(arrays['prefix_timing_metrics_valid'][0])
                self.assertTrue(np.isnan(arrays['prefix_timestamp_mismatches'][0]))
                self.assertTrue(np.isnan(arrays['prefix_max_absolute_time_difference_ms'][0]))
                self.assertIsNone(report['comparison']['prefix']['timestamp_mismatches'])
            else:
                self.assertTrue(arrays['prefix_timing_metrics_valid'][0])
                self.assertEqual(arrays['prefix_timestamp_mismatches'][0], 1)
                self.assertEqual(arrays['prefix_max_absolute_time_difference_ms'][0], 0.25)

    def test_finite_observed_initial_state_difference_is_diagnostic(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cortical-state-') as temporary:
            report, arrays = self.execute(Path(temporary), 'initial_vm')
        self.assertEqual(report['exit_code'], 2)
        self.assertTrue(report['comparison']['prefix']['event_multisets_exact'])
        self.assertFalse(report['comparison']['initial_vm']['exact'])
        self.assertEqual(arrays['initial_vm_difference_mV'][1], 0.125)

    def test_invalid_bindings_source_inputs_and_identity_are_errors(self):
        for variant in ('graph', 'rng', 'input_binding', 'background', 'lgn_prefix', 'identity',
                        'incomplete_pair', 'helper_code', 'source_missing', 'implementation_binding'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(prefix='synthetic-cortical-invalid-') as temporary:
                root = Path(temporary)
                first, second, input_pair = fixtures(root, variant)
                with mock.patch.multiple(cortical, CORTICAL_UNITS=3, LGN_UNITS=2):
                    with self.assertRaises(ValueError):
                        cortical.compare_archives(first, second, input_pair, root/'output')
                report = json.loads((root/'output/report.json').read_text())
                self.assertEqual(report['exit_code'], 1)
                self.assertFalse(report['scientific_qualification'])
                self.assertEqual(report['new_cortical_executions'], 0)
                expected_error = {
                    'helper_code': 'CORTICAL_PAIR_NATIVE_SOURCE_HASH_DIFFERS:',
                    'source_missing': 'CORTICAL_PAIR_NATIVE_SOURCE_HASH_MISSING_OR_INVALID:',
                    'implementation_binding': 'CORTICAL_PAIR_NATIVE_IMPLEMENTATION_HASH_UNBOUND:',
                }.get(variant)
                if expected_error:
                    self.assertTrue(report['error'].startswith(expected_error), report['error'])


if __name__ == '__main__':
    unittest.main()
