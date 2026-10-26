"""Differential temporal-bin tests against the existing dense readout reference."""
import unittest
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import h5py
from . import pfc_temporal_readouts as temporal
from . import pfc_scale_analysis as native_scale
from . import test_pfc_scale_analysis as reference_fixture
from .provenance import sha256_file
import numpy as np

from .pfc_temporal_readouts import temporal_rows, compare_phase, delay_summary
from .pfc_response_analysis import summarize_counts
from .pfc_scale_analysis import scalar_readout
from .pfc_delay_analysis import _check_readout


class TemporalReadoutTests(unittest.TestCase):
    def angular_rows(self, angles):
        rows = []
        for index, angle in enumerate(angles):
            theta = np.deg2rad(angle) if angle is not None else 0.
            rows.append(dict(start_tick=index*400, stop_tick=(index+1)*400,
                excitatory_spikes=10 if angle is not None else 0,
                vector_x_spikes=10*np.cos(theta) if angle is not None else 0.,
                vector_y_spikes=10*np.sin(theta) if angle is not None else 0.,
                direction_defined=angle is not None, decoded_angle_deg=angle))
        return rows

    def test_lagged_angular_wrap_and_coherent_rotation(self):
        rows = self.angular_rows([179., -179., -177., -175.])
        result = temporal.lagged_angular_displacement(rows, [0,1600], bin_ticks=400, lag_ticks=[400,800])
        self.assertEqual(result[0]['candidate_pairs'],3)
        self.assertAlmostEqual(result[0]['mean_signed_displacement_deg'],2.)
        self.assertAlmostEqual(result[0]['mean_squared_displacement_deg2'],4.)
        self.assertAlmostEqual(result[1]['mean_signed_displacement_deg'],4.)
        self.assertAlmostEqual(result[1]['mean_squared_displacement_deg2'],16.)
        self.assertAlmostEqual(result[1]['mean_cosine_increment'],np.cos(np.deg2rad(4)))

    def test_lagged_angular_silent_endpoint_withholds_all_increments(self):
        result = temporal.lagged_angular_displacement(self.angular_rows([0,None,5]), [0,1200],
                    bin_ticks=400, lag_ticks=[400])[0]
        self.assertEqual(result['candidate_pairs'],2)
        self.assertEqual(result['undefined_pairs'],2)
        self.assertEqual(result['undefined_unique_endpoint_bins'],1)
        self.assertIsNone(result['mean_signed_displacement_deg'])
        self.assertIsNone(result['mean_cosine_increment'])
        self.assertEqual(result['missing_reason'],'undefined_endpoint_pairs')

    def test_lagged_angular_partial_bin_and_no_pair_are_explicit(self):
        result = temporal.lagged_angular_displacement(self.angular_rows([0,2,4,6]), [0,880,1600],
                    bin_ticks=400, lag_ticks=[400])
        self.assertEqual(result[0]['phase_whole_bins'],2)
        self.assertEqual(result[0]['phase_uncovered_ticks'],80)
        self.assertEqual(result[0]['candidate_pairs'],1)
        self.assertEqual(result[1]['phase_whole_bins'],1)
        self.assertEqual(result[1]['phase_uncovered_ticks'],320)
        self.assertEqual(result[1]['candidate_pairs'],0)
        self.assertEqual(result[1]['missing_reason'],'no_phase_contained_pairs')

    def test_lagged_angular_antipodal_principal_branch_and_bad_metadata(self):
        rows = self.angular_rows([0,180])
        result = temporal.lagged_angular_displacement(rows,[0,800],bin_ticks=400,lag_ticks=[400])[0]
        self.assertEqual(result['mean_signed_displacement_deg'],-180.)
        self.assertEqual(result['mean_squared_displacement_deg2'],32400.)
        rows[-1]['decoded_angle_deg']=170.
        with self.assertRaisesRegex(ValueError,'SAVED_DIRECTION_CHANGED'):
            temporal.lagged_angular_displacement(rows,[0,800],bin_ticks=400,lag_ticks=[400])
        rows[-1]['decoded_angle_deg']=180.
        rows[-1]['stop_tick']=801
        with self.assertRaises(ValueError):
            temporal.lagged_angular_displacement(rows,[0,800],bin_ticks=400,lag_ticks=[400])

    def test_native_bin_edges_empty_bins_and_dense_reference(self):
        ticks = np.array([0, 0, 99, 100, 199, 299], dtype=np.int64)
        ids = np.array([0, 8, 3, 3, 9, 6], dtype=np.int64)
        rows = temporal_rows(ticks, ids, n_e=8, n_i=2, dt_ms=.05, bin_ms=5., duration_ms=20.)
        self.assertEqual(len(rows), 4)
        for row in rows:
            selected = ids[(ticks >= row['start_tick']) & (ticks < row['stop_tick'])]
            dense = np.bincount(selected, minlength=10)
            _check_readout(row, summarize_counts(dense, 8, 120., 30., 5.))
        self.assertEqual(rows[-1]['excitatory_spikes']+rows[-1]['inhibitory_spikes'], 0)
        self.assertIsNone(rows[-1]['decoded_angle_deg'])
        self.assertIsNone(rows[-1]['resultant_length'])

    def test_phase_aggregation_preserves_signed_moments_and_rates(self):
        ticks = np.array([0, 2, 3, 100, 101, 200, 201], dtype=np.int64)
        ids = np.array([0, 0, 8, 3, 3, 5, 9], dtype=np.int64)
        rows = temporal_rows(ticks, ids, n_e=8, n_i=2, dt_ms=.05, bin_ms=5., duration_ms=15.)
        reference = dict(phase='complete', start_ms=0., stop_ms=15., **scalar_readout(ids, 8, 2, 15.))
        self.assertEqual(len(compare_phase(rows, reference)), 11)
        changed = dict(reference, vector_x_spikes=reference['vector_x_spikes']+1.)
        with self.assertRaisesRegex(ValueError, 'aggregation differs'):
            compare_phase(rows, changed)

    def test_isotropic_and_silent_directions_remain_undefined(self):
        rows = temporal_rows(np.zeros(8, dtype=np.int64), np.arange(8, dtype=np.int64),
                             n_e=8, n_i=2, dt_ms=.05, bin_ms=5., duration_ms=10.)
        self.assertFalse(rows[0]['direction_defined'])
        self.assertIsNone(rows[0]['decoded_angle_deg'])
        self.assertFalse(rows[1]['direction_defined'])

    def test_invalid_order_clock_identity_and_terminal_event_are_rejected(self):
        for ticks, ids, duration in (([100, 0], [0, 1], 10.), ([200], [0], 10.),
                                     ([0], [10], 10.), ([0], [0], 7.)):
            with self.subTest(ticks=ticks, ids=ids, duration=duration), self.assertRaises(ValueError):
                temporal_rows(np.asarray(ticks), np.asarray(ids), n_e=8, n_i=2,
                              dt_ms=.05, bin_ms=5., duration_ms=duration)

    def test_delay_summary_excludes_cue_and_keeps_undefined_directions(self):
        rows = [dict(start_ms=95, stop_ms=100, cue_selectivity_hz=99,
                     circular_cue_error_deg=179, excitatory_spikes=2, inhibitory_spikes=3),
                dict(start_ms=100, stop_ms=105, cue_selectivity_hz=-1,
                     circular_cue_error_deg=None, excitatory_spikes=0, inhibitory_spikes=0),
                dict(start_ms=105, stop_ms=110, cue_selectivity_hz=3,
                     circular_cue_error_deg=-2, excitatory_spikes=4, inhibitory_spikes=5)]
        summary = delay_summary(rows, 100.)
        self.assertEqual(summary['delay_bins'], 2)
        self.assertEqual(summary['cue_selectivity_mean_hz'], 1.)
        self.assertEqual(summary['cue_selectivity_min_hz'], -1.)
        self.assertEqual(summary['excitatory_spikes'], 4)
        self.assertEqual(summary['direction_defined_bins'], 1)
        self.assertEqual(summary['max_abs_circular_cue_error_deg'], 2.)


class NativeReferenceTemporalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pfc_native_temporal_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/'archive'
        self.root.mkdir()

    def add_terminal_event(self):
        case_id = 'pfc_93001_cue'
        reports = {}
        for role in ('native', 'factorized', 'factorized_replay'):
            report_path = self.root/role/'report.json'
            report = native_scale.read(report_path)
            case = next(row for row in report['cases'] if row['case_id'] == case_id)
            raw_path = self.root/role/case['raw_file']
            with np.load(raw_path, allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            arrays['spike_ticks'] = np.r_[arrays['spike_ticks'], np.int64(100000)]
            arrays['spike_unit_indices'] = np.r_[arrays['spike_unit_indices'], np.int64(0)]
            arrays['spike_seconds'] = np.r_[arrays['spike_seconds'], 5.]
            arrays['native_spike_counts'][0] += 1
            np.savez_compressed(raw_path, **arrays)
            case.update(native_events=case['native_events']+1, raw_sha256=sha256_file(raw_path))
            native_scale.write_json(report_path, report)
            reports[role] = report
            if role == 'native':
                catalog_path = self.root/'native/dataset/dataset.json'
                catalog = native_scale.read(catalog_path)
                shard = catalog['shards'][0]
                shard_path = self.root/'native/dataset'/shard['path']
                with h5py.File(shard_path, 'r+') as stored:
                    group = stored[case_id]
                    for key in ('spike_ticks', 'spike_unit_indices'):
                        del group[key]
                        group.create_dataset(key, data=arrays[key])
                    metadata = json.loads(group.attrs['metadata_json'])
                    metadata['source_record']['raw_sha256'] = case['raw_sha256']
                    group.attrs['metadata_json'] = json.dumps(metadata)
                shard['sha256'] = sha256_file(shard_path)
                next(row for row in shard['episodes'] if row['episode_id'] == case_id)['event_count'] += 1
                native_scale.write_json(catalog_path, catalog)
        comparison = native_scale.read(self.root/'source_comparison.json')
        comparison['candidate_report_sha256'] = sha256_file(self.root/'factorized/report.json')
        comparison['reference_bindings'][0]['report_sha256'] = sha256_file(self.root/'native/report.json')
        comparison['array_values'] += 3
        next(row for row in comparison['cases'] if row['case_id'] == case_id)['values'] += 3
        native_scale.write_json(self.root/'source_comparison.json', comparison)
        replay = native_scale.read(self.root/'factorized_replay.json')
        replay.update(original_report_sha256=sha256_file(self.root/'factorized/report.json'),
                      replayed_report_sha256=sha256_file(self.root/'factorized_replay/report.json'),
                      fresh_process_array_values_checked=replay['fresh_process_array_values_checked']+3)
        native_scale.write_json(self.root/'factorized_replay.json', replay)

    def refresh_delivery(self):
        phase = native_scale.read(self.root/'analysis/analysis.json')
        files = [dict(path=str(path.relative_to(self.root)), bytes=path.stat().st_size,
                      sha256=sha256_file(path)) for path in sorted(self.root.rglob('*'))
                 if path.is_file() and path != self.root/'delivery.json']
        native_scale.write_json(self.root/'delivery.json', dict(
            status='ARCHIVED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC',
            complete_source_episodes=9, units=2560, duration_ms=5000,
            source_rng_seeds=list(native_scale.SOURCE_SEEDS),
            source_revision=native_scale.SOURCE_REVISION, source_entry_sha256=native_scale.SOURCE_ENTRY_SHA256,
            production_episodes=0, scientific_fidelity_qualified=False,
            full_scale_qualified=False, working_memory_qualified=False,
            native_report_sha256=sha256_file(self.root/'native_recovery/report.json'),
            source_comparison_sha256=sha256_file(self.root/'source_comparison.json'),
            fresh_replay_sha256=sha256_file(self.root/'factorized_replay.json'),
            analysis_sha256=sha256_file(self.root/'analysis/analysis.json'), files=files))

    def make_archive(self, terminal=False):
        reference_fixture.make_fixture(self.root)
        if terminal:
            self.add_terminal_event()
        native_scale.analyze(self.root/'native', self.root/'factorized', self.root/'factorized_replay',
            self.root/'source_comparison.json', self.root/'factorized_replay.json', self.root/'design.json',
            self.root/'native.yaml', self.root/'factorized.yaml', self.root/'analysis')
        (self.root/'native').rename(self.root/'native_recovery')
        (self.root/'native').mkdir()
        partial = native_scale.read(self.root/'native_recovery/report.json')
        partial.update(status='INTERRUPTED', cases=partial['cases'][:3])
        native_scale.write_json(self.root/'native/report.json', partial)
        configs = self.root/'Libs/configs'
        configs.mkdir(parents=True)
        shutil.copyfile(self.root/'native.yaml', configs/'neural_population_pfc_reference_size.yaml')
        shutil.copyfile(self.root/'factorized.yaml', configs/'neural_population_pfc_reference_size_factorized.yaml')
        code = self.root/'Libs/population_dynamics'
        code.mkdir()
        for module in (native_scale, native_scale.pfc_source_comparison, native_scale.pfc_reference_replay):
            shutil.copyfile(module.__file__, code/Path(module.__file__).name)
        self.refresh_delivery()

    def test_real_cli_retains_all_nine_cases_and_existing_summary_interface(self):
        self.make_archive()
        output = Path(self.temporary.name)/'temporal'
        result = subprocess.run([sys.executable, '-m', temporal.__name__,
            '--native-reference-archive', str(self.root), '--output', str(output)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = native_scale.read(output/'analysis.json')
        self.assertEqual(record['status'], 'COMPLETE_SOURCE_TEMPORAL_READOUTS')
        self.assertEqual(record['native_report_path'], 'native_recovery/report.json')
        self.assertEqual((len(record['completed_cases']), record['bins']), (9, 9000))
        self.assertEqual(record['phase_linear_statistic_comparisons'], 495)
        self.assertEqual(record['terminal_tick_events_excluded'], 0)
        self.assertEqual(record['current_observation_status'], 'UNAVAILABLE_LEGACY_ARCHIVE')
        self.assertFalse(record['scientific_fidelity_qualified'])
        self.assertEqual(record['new_source_episodes'], 0)
        summary = Path(self.temporary.name)/'summary'
        temporal.summarize_existing(output, summary)
        summarized = native_scale.read(summary/'summary.json')
        self.assertEqual(len(summarized['cases']), 9)
        self.assertEqual(summarized['delay_boundary_semantics'], record['delay_boundary_semantics'])
        self.assertEqual(summarized['current_observation_status'], 'UNAVAILABLE_LEGACY_ARCHIVE')
        self.assertEqual(summarized['coupled_core_units'], 2560)

    def test_changed_native_provenance_and_rebound_missing_phase_are_rejected(self):
        self.make_archive()
        delivery = native_scale.read(self.root/'delivery.json')
        native_scale.write_json(self.root/'delivery.json', dict(delivery, native_report_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'unique complete native report'):
            temporal.analyze_native_reference(self.root, Path(self.temporary.name)/'bad_native')
        native_scale.write_json(self.root/'delivery.json', delivery)
        path = self.root/'analysis/responses.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        path.unlink()
        native_scale.write_rows(path, rows[1:])
        phase = native_scale.read(self.root/'analysis/analysis.json')
        replay = native_scale.read(self.root/'analysis/readout_replay.json')
        phase['artifacts']['responses.csv'] = sha256_file(path)
        replay['artifacts'] = phase['artifacts']
        native_scale.write_json(self.root/'analysis/readout_replay.json', replay)
        phase['readout_replay_sha256'] = sha256_file(self.root/'analysis/readout_replay.json')
        native_scale.write_json(self.root/'analysis/analysis.json', phase)
        self.refresh_delivery()
        with self.assertRaisesRegex(ValueError, 'complete native reference phase coverage'):
            temporal.analyze_native_reference(self.root, Path(self.temporary.name)/'missing_phase')

    def test_historical_replay_identity_is_bound_to_archived_code(self):
        self.make_archive()
        code = self.root/'Libs/population_dynamics/pfc_reference_replay.py'
        code.write_text(code.read_text()+'\n# Distinct historical implementation snapshot.\n')

        def rebind(replay_hash):
            replay_path = self.root/'factorized_replay.json'
            replay = native_scale.read(replay_path)
            replay['comparison_implementation_sha256'] = replay_hash
            native_scale.write_json(replay_path, replay)
            phase_path = self.root/'analysis/analysis.json'
            phase = native_scale.read(phase_path)
            phase['fresh_replay_sha256'] = sha256_file(replay_path)
            native_scale.write_json(phase_path, phase)
            self.refresh_delivery()

        rebind(sha256_file(code))
        record = temporal.analyze_native_reference(self.root, Path(self.temporary.name)/'historical')
        self.assertEqual(record['status'], 'COMPLETE_SOURCE_TEMPORAL_READOUTS')
        self.assertEqual(record['consumed_manifest_files']['Libs/population_dynamics/pfc_reference_replay.py'],
                         sha256_file(code))
        rebind('0'*64)
        with self.assertRaisesRegex(ValueError, 'complete fresh factorized replay binding'):
            temporal.analyze_native_reference(self.root, Path(self.temporary.name)/'wrong_code')

    def test_terminal_tick_events_are_preserved_as_explicit_exclusions(self):
        self.make_archive(terminal=True)
        output = Path(self.temporary.name)/'terminal'
        record = temporal.analyze_native_reference(self.root, output)
        self.assertEqual(record['terminal_tick_events_excluded'], 1)
        case = next(row for row in record['completed_cases'] if row['case_id'] == 'pfc_93001_cue')
        self.assertEqual(case['archived_events']-case['events'], 1)
        self.assertEqual(case['terminal_tick_events_excluded'], 1)
        self.assertEqual(record['bins'], 9000)

    def test_modes_are_exclusive_and_original_dispatch_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'cannot combine'):
            temporal.main(['--native-reference-archive', str(self.root), '--summary-of', str(self.root),
                           '--output', str(self.root/'unused')])
        with patch.object(temporal, 'analyze') as streamed:
            temporal.main(['--config', 'config.yaml', '--source-run', 'source',
                           '--reference-readouts', 'reference', '--output', 'out'])
            streamed.assert_called_once_with(Path('config.yaml'), [Path('source')],
                                              Path('reference'), Path('out'))
        with patch.object(temporal, 'summarize_existing') as summary:
            temporal.main(['--summary-of', 'source', '--output', 'out'])
            summary.assert_called_once_with(Path('source'), Path('out'))


if __name__ == '__main__':
    unittest.main()
