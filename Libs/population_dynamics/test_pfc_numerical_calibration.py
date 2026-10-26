"""Coverage and authority regressions for declared PFC calibration replication."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from . import pfc_numerical_calibration as preparation
from . import pfc_scale_analysis as phases
from . import pfc_temporal_readouts as temporal
from . import pfc_fluctuation_analysis as fluctuations
from . import pfc_source_dataset as dataset
from . import test_pfc_source_scale_analysis as fixtures


class CalibrationReplicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SourceReadoutTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.seeds = [95001, 95002, 95003, 95004]
        config = self.fixture.config
        config.update(readout_bin_ms=5., excitatory_preference_bins=16,
            source_replication=dict(purpose='numerical_calibration',
                independent_unit='joint_initial_background_rng_block', joint_rng_seeds=self.seeds),
            cases=[dict(case_id=f'pfc_{seed}_{condition}', source_rng_seed=seed,
                        stimulus_strength_nA=strength, distractor_strength_nA=0.)
                   for seed in self.seeds for condition, strength in phases.CONDITIONS.items()])
        self.save()
        _, self.fixture.registered, self.fixture.seeds, self.fixture.dimensions = phases._source_registration(self.fixture.config_path)

    def save(self):
        self.fixture.config_path.write_text(yaml.safe_dump(self.fixture.config))

    def test_legacy_more_than_three_blocks_requires_explicit_declaration(self):
        self.fixture.config.pop('source_replication'); self.save()
        with self.assertRaisesRegex(ValueError, 'nine source cases'):
            phases._source_registration(self.fixture.config_path)

    def test_malformed_replication_or_missing_control_is_rejected(self):
        original = copy.deepcopy(self.fixture.config)
        variants = []
        for seeds in ([95001, 95001, 95003, 95004], [True, 95002, 95003, 95004],
                      [95001, 95002, 95003, 2**32], [95002, 95001, 95003, 95004]):
            v = copy.deepcopy(original); v['source_replication']['joint_rng_seeds'] = seeds; variants.append(v)
        v = copy.deepcopy(original); v['cases'].pop(); variants.append(v)
        v = copy.deepcopy(original); v['cases'][0]['stimulus_strength_nA'] = .5; variants.append(v)
        v = copy.deepcopy(original); v['source_replication']['purpose'] = 'numerical_confirmation'; variants.append(v)
        for value in variants:
            with self.subTest(value=value.get('source_replication')):
                self.fixture.config = value; self.save()
                with self.assertRaises(ValueError): phases._source_registration(self.fixture.config_path)

    def test_phase_temporal_fluctuation_and_dataset_keep_every_block(self):
        sources = [self.fixture.add_current(self.fixture.archive(case)) for case in self.fixture.registered]
        reference = self.root/'reference'; reference.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            report = phases.analyze_source_runs(sources, self.fixture.config_path, reference/'analysis')
        self.assertEqual(report['complete_source_episodes'], 12)
        self.assertEqual(report['registered_source_rng_blocks'], 4)
        (reference/'config.yaml').write_bytes(self.fixture.config_path.read_bytes())
        self.fixture.write_json(reference/'input_origins.json', [
            dict(case_id=b['case_id'], delivery_sha256=b['delivery_sha256']) for b in report['source_bindings']])
        names = ['config.yaml', 'input_origins.json']+['analysis/'+name for name in
            ('analysis.json', 'readout_replay.json', 'responses.csv', 'phase_unit_counts.npz')]
        self.fixture.write_json(reference/'delivery.json', dict(status='ARCHIVED_PFC_SOURCE_READOUTS',
            exit_code=0, files=[self.fixture.file_row(reference, name) for name in names]))
        measurements = self.root/'measurements.yaml'
        measurements.write_text(yaml.safe_dump(dict(scope='development_calibration', measurements=dict(
            observation_bin_ms=5., phases=list(phases.PHASES), all_unit_count_windows_ms=[5,50,100]))))
        with contextlib.redirect_stdout(io.StringIO()):
            result = temporal.analyze(self.fixture.config_path, sources, reference, self.root/'temporal')
            temporal.summarize_existing(self.root/'temporal', self.root/'summary')
            fluctuation = fluctuations.run(sources, self.fixture.config_path, reference, self.root/'fluctuation', measurement_protocol=measurements)
            exported = dataset.export(sources, self.fixture.config_path, self.root/'dataset')
        self.assertEqual(result['bins'], 12000); self.assertEqual(result['phase_linear_statistic_comparisons'], 660)
        self.assertEqual(len(phases.read(self.root/'summary/summary.json')['cases']), 12)
        self.assertEqual(fluctuation['joint_rng_blocks'], self.seeds)
        self.assertEqual(fluctuation['unit_count_histograms']['unit_window_observations'], 138000)
        with np.load(self.root/'fluctuation/unit_count_histograms.npz') as values:
            self.assertEqual(values['expected_observations'].shape, (3,4,5,3,17))
        self.assertEqual(fluctuation['unit_phase_count_values_verified'], 12*5*20)
        self.assertEqual(fluctuation['field_bin_count_values_verified'], 12*1000*17)
        with np.load(self.root/'fluctuation/field_fluctuations.npz') as values:
            self.assertEqual(values['cue_field_counts'].shape, (4, 1000, 17))
        self.assertEqual(exported['complete_episodes'], 12)
        self.assertEqual(exported['registered_source_cases'], 12)
        self.assertEqual(exported['production_episodes'], 0)
        archive = dataset.CircuitDataset(self.root/'dataset/dataset', splits=('development',))
        self.assertEqual(len(archive.episodes), 12)
        self.assertEqual(len(archive.catalog['circuits']), 1)
        self.assertEqual(len({row['block_id'] for row in archive.episodes.values()}), 4)
        with self.assertRaisesRegex(ValueError, 'registered calibration cases'):
            fluctuations.run(sources[:-1], self.fixture.config_path, reference, self.root/'missing')


class CalibrationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = Path(__file__).resolve().parents[2]
        self.protocol_path = self.project/'Libs/configs/neural_population_pfc_numerical_calibration.yaml'

    def test_real_preparation_preserves_all32_blocks_and_reserved_confirmation(self):
        report = preparation.prepare(self.protocol_path, self.root/'prepared', self.project)
        self.assertEqual(report['expected_complete_source_episodes'], 288)
        self.assertEqual(report['independent_joint_blocks'], 32)
        self.assertFalse(report['confirmation_execution_released'])
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertEqual(report['new_source_simulations'], 0)
        for clock in report['clocks']:
            config, registered, seeds, _ = phases._source_registration(self.root/'prepared'/clock['config'])
            self.assertEqual(len(registered), 96)
            self.assertEqual(seeds, list(range(95001, 95033)))
            selected = [case for batch in clock['batches'] for case in batch['case_ids']]
            self.assertEqual(selected, list(registered))
            self.assertEqual(len(set(selected)), 96)
            self.assertEqual(config['excitatory_neurons']+config['inhibitory_neurons'], 100000)

    def test_overlap_with_inspected_or_confirmation_and_changed_clock_rejected(self):
        original = yaml.safe_load(self.protocol_path.read_text())
        for key in ('previously_inspected_joint_rng_seeds', 'confirmation_seed_reservation', 'source_configs'):
            value = copy.deepcopy(original)
            if key == 'source_configs': value[key][1]['dt_ms'] = .05
            else: value[key].append(value['joint_rng_seeds'][0])
            path = self.root/(key+'.yaml'); path.write_text(yaml.safe_dump(value))
            with self.subTest(key=key), self.assertRaises(ValueError):
                preparation.prepare(path, self.root/('output_'+key), self.project)


if __name__ == '__main__':
    unittest.main()
