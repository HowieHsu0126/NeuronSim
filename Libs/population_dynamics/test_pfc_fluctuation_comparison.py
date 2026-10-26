"""Signed lag covariance and complete delivered-cohort regressions."""
from contextlib import contextmanager
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import yaml
from . import pfc_fluctuation_comparison as analysis
from .provenance import sha256_file


def file_row(root, name):
    path = root/name
    return dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))


def bind(root):
    report = analysis.read(root/'analysis/report.json')
    report['config_sha256'] = sha256_file(root/'config.yaml')
    names=['unit_moments.npz','field_fluctuations.npz','population_variance.csv']
    if (root/'analysis/unit_count_histograms.npz').exists():names.append('unit_count_histograms.npz')
    report['outputs']={name:file_row(root,'analysis/'+name) for name in names}
    analysis.write_json(root/'analysis/report.json', report)
    delivery = analysis.read(root/'delivery.json')
    delivery['files'] = [file_row(root,name) for name in
        ['config.yaml','analysis/report.json']+['analysis/'+name for name in names]]
    if (root/'reference/delivery.json').exists():
        delivery['files'] += [file_row(root,'reference/'+name) for name in analysis.pfc_phase_responses.REFERENCE_FILES]
    analysis.write_json(root/'delivery.json', delivery)


def make_cohort(root, dt, native=False, seeds=None):
    """Keep all units; one active unit per field keeps synthetic fixtures small."""
    (root/'analysis').mkdir(parents=True)
    config = yaml.safe_load((Path(__file__).parents[1]/'configs/neural_population_pfc_reference_size.yaml').read_text())
    n_e, n_i = (2048, 512) if native else (80000, 20000)
    seeds = seeds if seeds is not None else ([93001, 93002, 93003] if native else [94001, 94002, 94003])
    config.update(excitatory_neurons=n_e, inhibitory_neurons=n_i, weight_scaling_factor=2048/n_e,
        dt_ms=dt, source_rng_seed=seeds[0], cases=[
        dict(case_id=f'pfc_{seed}_{condition}', source_rng_seed=seed,
             stimulus_strength_nA=strength, distractor_strength_nA=0.)
        for seed in seeds for condition, strength in (('cue', .06), ('omission', 0.), ('strong_cue', .5))])
    if len(seeds) != 3:
        config['synapse_execution']='factorized_uniform_projections'
        config['source_replication']=dict(purpose='numerical_calibration',
            independent_unit='joint_initial_background_rng_block',joint_rng_seeds=seeds)
    (root/'config.yaml').write_text(yaml.safe_dump(config))
    mapping, sizes = analysis.unit_mapping(n_e, n_i, 16)
    active = np.array([np.flatnonzero(mapping == group)[0] for group in range(17)])
    units = dict(unit_ids=np.arange(n_e+n_i), unit_field_indices=mapping)
    fields = dict(field_unit_counts=sizes, source_rng_seeds=np.array(seeds))
    rows = []
    for condition in analysis.CONDITIONS:
        counts = (4+np.arange(1000)[None, :, None] % 7
                  +np.arange(len(seeds))[:, None, None]*(1+np.arange(17)[None, None, :] % 3))
        if condition == 'omission': counts = np.zeros_like(counts)
        counts[:,:20] = 0  # All paired controls have the same pre-cue baseline.
        fields[condition+'_field_counts'] = counts
        means = np.zeros((5, n_e+n_i)); variances = np.zeros_like(means); covariances = []
        for index, (phase, (start, stop)) in enumerate(analysis.PHASES.items()):
            selected = counts[:, round(start/5):round(stop/5)]
            means[index, active] = selected.mean(axis=(0, 1))
            variances[index, active] = selected.var(axis=0, ddof=1).mean(axis=0)
            centered = selected-selected.mean(axis=0, keepdims=True)
            covariances.append(np.einsum('bti,btj->ij', centered, centered)/((len(seeds)-1)*selected.shape[1]))
            for population, mask in (('E', mapping<16), ('I', mapping==16), ('all', mapping>=0)):
                groups = np.unique(mapping[mask]); number = int(mask.sum())
                mean = means[index, mask].mean(); variance = variances[index, mask].mean()
                pv = (selected[:, :, groups].sum(axis=2)/number).var(axis=0, ddof=1).mean()
                rows.append(dict(condition=condition, phase=phase, population=population, units=number,
                    joint_rng_blocks=len(seeds), bin_ms=5., time_bins=selected.shape[1], mean_count_per_unit=mean,
                    mean_single_unit_count_variance=variance, population_mean_count_variance=pv,
                    diagonal_contribution_to_population_variance=variance/number,
                    ordered_offdiagonal_contribution_to_population_variance=pv-variance/number,
                    pooled_single_unit_ensemble_fano=None if mean == 0 else variance/mean))
        fields[condition+'_field_covariance_counts'] = np.stack(covariances)
        units[condition+'_mean_counts'] = means; units[condition+'_variance_counts'] = variances
    np.savez_compressed(root/'analysis/unit_moments.npz', **units)
    np.savez_compressed(root/'analysis/field_fluctuations.npz', **fields)
    analysis.write_rows(root/'analysis/population_variance.csv', rows)
    analysis.write_json(root/'analysis/report.json', dict(status='COMPLETE_SOURCE_FLUCTUATION_CALIBRATION',
        completed_source_cases=len(config['cases']), registered_source_cases=len(config['cases']), coupled_core_units=n_e+n_i, dt_ms=dt,
        joint_rng_blocks=seeds, phases_ms=analysis.PHASES, bin_ms=5., field_unit_counts=sizes.tolist(),
        source_bindings=[dict(case_id=row['case_id'],report_sha256='fixture-report-'+row['case_id'],
            delivery_sha256='fixture-delivery-'+row['case_id'],source_runner_sha256='fixture-runner') for row in config['cases']],
        common_execution_binding=dict(backend='numpy', code=dict(executor_sha256='fixture-engine', drive_sha256='fixture-drive')),
        source_backend='native_brian2' if native else 'numpy', confidence_intervals=False, p_values=False,
        q_values=False, scientific_fidelity_qualified=False))
    analysis.write_json(root/'delivery.json', dict(status='ARCHIVED_PFC_SOURCE_READOUTS', exit_code=0,
        readout_mode='reference-fluctuations' if native else 'fluctuations'))
    bind(root)



def add_histograms(root):
    """Reconstruct fixture counts: one active unit and remaining silent units per field."""
    report=analysis.read(root/'analysis/report.json')
    with np.load(root/'analysis/field_fluctuations.npz') as stored:
        fields={key:stored[key] for key in stored.files}
    sizes=fields['field_unit_counts']; seeds=fields['source_rng_seeds'].tolist()
    windows=[5,50,100]; histograms={}
    for ci,condition in enumerate(analysis.CONDITIONS):
        for pi,(start,stop) in enumerate(analysis.PHASES.values()):
            values=fields[condition+'_field_counts'][:,round(start/5):round(stop/5)]
            for wi,width in enumerate(windows):
                analysis.pfc_count_histograms.accumulate(histograms,values,np.arange(17),round(width/5),
                    condition=ci,phase=pi,window=wi)
                for bi in range(len(seeds)):
                    for fi,size in enumerate(sizes):
                        key=(ci,bi,pi,wi,fi,0)
                        histograms[key]=histograms.get(key,0)+(int(size)-1)*round((stop-start)/width)
    report['unit_count_histograms']=analysis.pfc_count_histograms.write(root/'analysis/unit_count_histograms.npz',
        histograms,conditions=analysis.CONDITIONS,seeds=seeds,phases=analysis.PHASES,windows_ms=windows,field_sizes=sizes)
    analysis.write_json(root/'analysis/report.json',report)
    bind(root)


class LagCovarianceTests(unittest.TestCase):
    def test_orientation_negative_diagonal_and_last_defined_lag(self):
        x = np.array([[[9, 8], [7, 11], [10, 6]], [[10, 10], [10, 10], [10, 10]],
                      [[11, 12], [13, 9], [10, 14]]])
        for lag, expected in enumerate(([[10/3, -1/3], [-1/3, 7]],
                                        [[1.5, 5.5], [3., -3.]], [[0., 4.], [0., 8.]])):
            np.testing.assert_allclose(analysis.lag_covariance(x, lag), expected, atol=1e-14)
        self.assertIsNone(analysis.lag_covariance(x, 3))
        self.assertEqual(analysis.difference_summary(None, None)['status'], 'UNDEFINED_PHASE_CONTAINED_LAG')

    def test_shared_deterministic_transient_removed(self):
        x = np.repeat(np.array([[[1, 3], [20, 0], [4, 30]]]), 3, axis=0)
        for lag in (0, 1, 2):
            np.testing.assert_array_equal(analysis.lag_covariance(x, lag), np.zeros((2, 2)))

    def test_sample_denominator_with_one_time_bin(self):
        np.testing.assert_array_equal(analysis.lag_covariance(np.array([[[0.]], [[2.]]]), 0), [[2.]])

    def test_invalid_lag_and_nonfinite_or_missing_blocks_rejected(self):
        good = np.ones((3, 4, 2))
        for values, lag in ((good, -1), (good, 1.), (good, True), (good[:1], 0), (good*np.nan, 0)):
            with self.assertRaises(ValueError): analysis.lag_covariance(values, lag)


class FluctuationComparisonArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.clocks = [cls.root/f'clock_{index}' for index in range(3)]
        for path, dt in zip(cls.clocks, analysis.CLOCKS): make_cohort(path, dt)
        cls.native = cls.root/'native'; make_cohort(cls.native, .05, native=True)
        cls.settings = cls.root/'comparison.yaml'
        cls.settings.write_text(yaml.safe_dump(dict(scope='development_calibration', bin_ms=5.,
            lag_ms=[0, 5, 10, 100], lag_scope='complete_within_phase_pairs', inference='descriptive_only')))

    @contextmanager
    def changed(self, root, name, change, *, rebind=True):
        saved = {key: (root/key).read_bytes() for key in set((name, 'analysis/report.json', 'delivery.json'))}
        try:
            change(root/name)
            if rebind: bind(root)
            yield
        finally:
            for key, value in saved.items(): (root/key).write_bytes(value)

    def command(self, output, settings=None):
        command = [sys.executable, '-B', '-m', analysis.__name__, '--config', str(settings or self.settings),
                   '--native-reference', str(self.native), '--output', str(output)]
        for path in self.clocks: command += ['--clock-archive', str(path)]
        return command

    def test_real_cli_complete_rows_units_and_size_normalization(self):
        output = self.root/'cli'
        result = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = analysis.read(output/'report.json')
        self.assertEqual(report['row_counts'], {'by_cohort_population_variance.csv': 180,
            'population_differences.csv': 810, 'unit_moment_clock_differences.csv': 60,
            'lagged_field_covariance_differences.csv': 180})
        self.assertEqual(report['undefined_phase_lag_rows'], 9); self.assertEqual(report['source_episodes_reused'], 36)
        for key in ('scientific_fidelity_qualified', 'finite_size_scaling_qualified',
                    'distributional_convergence_qualified', 'confidence_intervals', 'p_values', 'q_values'):
            self.assertIs(report[key], False)
        with (output/'unit_moment_clock_differences.csv').open() as stream: rows = list(csv.DictReader(stream))
        self.assertTrue(all('native' not in row['left_cohort']+row['right_cohort'] for row in rows))
        self.assertTrue(all(int(row['compared_units']) == 100000 for row in rows))
        with (output/'population_differences.csv').open() as stream: rows = list(csv.DictReader(stream))
        self.assertTrue(all(row['status'] == 'UNDEFINED' and row['left_minus_right'] == ''
            for row in rows if row['condition'] == 'omission' and row['metric'] == 'pooled_single_unit_ensemble_fano'))
        with np.load(output/'lagged_field_covariance_hz2.npz') as stored:
            self.assertEqual(len(stored.files), 228)
            x = stored['native_2560_cue_late_delay_lag_0ms'][0, 0]
            y = stored['numpy_dt_0p05_cue_late_delay_lag_0ms'][0, 0]
            self.assertAlmostEqual(x/y, (5000/128)**2)

    def test_duplicate_or_missing_registered_cases_rejected(self):
        root = self.clocks[0]
        for duplicate in (True, False):
            def change(path):
                value = analysis.read(path); value['source_bindings'] = value['source_bindings'][:-1]
                if duplicate: value['source_bindings'].append(value['source_bindings'][0])
                analysis.write_json(path, value)
            with self.subTest(duplicate=duplicate), self.changed(root, 'analysis/report.json', change):
                with self.assertRaisesRegex(ValueError, 'unique registered'): analysis.load_cohort(root)

    def test_changed_unit_map_or_integer_identity_dtype_rejected(self):
        root = self.clocks[0]
        for key in ('unit_ids', 'unit_field_indices'):
            def change(path):
                with np.load(path) as stored: values = {name: stored[name] for name in stored.files}
                values[key] = values[key].astype(float) if key == 'unit_ids' else values[key][::-1]
                np.savez_compressed(path, **values)
            with self.subTest(key=key), self.changed(root, 'analysis/unit_moments.npz', change):
                with self.assertRaisesRegex(ValueError, 'unit identities'): analysis.load_cohort(root)

    def test_changed_shared_settings_fail_cli_with_failed_report(self):
        root = self.clocks[1]
        def change(path):
            config = yaml.safe_load(path.read_text()); config['stimulus_width_deg'] = 31.
            path.write_text(yaml.safe_dump(config))
        with self.changed(root, 'config.yaml', change):
            output = self.root/'changed_settings'
            result = subprocess.run(self.command(output), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0); self.assertIn('only clock may differ', result.stderr)
            self.assertEqual(analysis.read(output/'report.json')['status'], 'FAILED')

    def test_changed_delivered_file_rejected_before_use(self):
        root = self.clocks[0]
        def change(path): path.write_bytes(path.read_bytes()+b'\n')
        with self.changed(root, 'analysis/population_variance.csv', change, rebind=False):
            with self.assertRaisesRegex(ValueError, 'size or hash mismatch'): analysis.load_cohort(root)

    def test_changed_zero_lag_covariance_rejected(self):
        root = self.clocks[0]
        def change(path):
            with np.load(path) as stored: values = {key: stored[key] for key in stored.files}
            values['cue_field_covariance_counts'][0, 0, 0] += 1
            np.savez_compressed(path, **values)
        with self.changed(root, 'analysis/field_fluctuations.npz', change):
            with self.assertRaisesRegex(ValueError, 'zero-lag covariance'):
                analysis.compare(self.clocks, self.native, self.settings, self.root/'changed_covariance')

    def test_duplicate_clock_and_unregistered_lag_scope_rejected(self):
        with self.assertRaisesRegex(ValueError, 'distinct registered clocks'):
            analysis.compare([self.clocks[0]]*3, self.native, self.settings, self.root/'duplicate_clocks')
        settings = self.root/'wrong_scope.yaml'; value = yaml.safe_load(self.settings.read_text())
        value['lag_scope'] = 'cross_phase_pairs'; settings.write_text(yaml.safe_dump(value))
        with self.assertRaisesRegex(ValueError, 'lag grid'):
            analysis.compare(self.clocks, self.native, settings, self.root/'wrong_scope')


    def test_count_cdfs_retain_every_block_and_verify_conservation(self):
        settings = self.root/'with_counts.yaml'
        value = yaml.safe_load(self.settings.read_text()); value['count_windows_ms'] = [5,50,100]
        settings.write_text(yaml.safe_dump(value))
        with self.assertRaisesRegex(ValueError,'complete declared count histograms'):
            analysis.compare(self.clocks,self.native,settings,self.root/'missing_counts')
        for root in self.clocks:
            add_histograms(root)
        report = analysis.compare(self.clocks,self.native,settings,self.root/'with_counts')
        self.assertEqual(report['count_cdf_comparison']['unit_window_mass_checks'],3*3*3*5*3*17)
        self.assertEqual(report['row_counts']['count_cdf_descriptive_distances.csv'],2*3*5*3*17)
        self.assertFalse(report['distributional_convergence_qualified'])
        with np.load(self.root/'with_counts/block_count_cdfs.npz') as stored:
            self.assertEqual(stored['numpy_dt_0p05_cdf'].shape[:5],(3,3,5,3,17))
            np.testing.assert_array_equal(stored['numpy_dt_0p05_cdf'],stored['numpy_dt_0p025_cdf'])
            np.testing.assert_array_equal(stored['source_rng_seeds'],[94001,94002,94003])
        for change_kind,message in (('mass','unit/window mass'),('spikes','spike totals')):
            def change(path):
                with np.load(path) as stored: payload={key:stored[key] for key in stored.files}
                if change_kind=='mass':
                    index=np.flatnonzero(payload['coordinates'][:,5]==0)[0]
                    payload['observations'][index]+=1
                else:
                    index=np.flatnonzero(payload['coordinates'][:,5]>0)[0]
                    payload['coordinates'][index,5]=payload['coordinates'][:,5].max()+1000
                np.savez_compressed(path,**payload)
            with self.subTest(change_kind=change_kind), self.changed(self.clocks[0],'analysis/unit_count_histograms.npz',change):
                with self.assertRaisesRegex(ValueError,message):
                    analysis.load_cohort(self.clocks[0])

    def test_signed_gapped_support_cdfs_average_blocks_equally(self):
        # Algebra fixture with unequal masses exercises equal block weighting.
        # Actual admitted source cohorts additionally require equal complete-window mass.
        examples = [({0:1,4:1},{2:6}), ({2:2},{0:1,2:4,4:1}), ({2:2},{2:6})]
        cohorts = {}
        shape = (3,2,5,1,1)
        for label,blocks in zip(('coarse','middle','fine'),examples):
            rows,mass = [],[]
            for ci in range(3):
                for bi,histogram in enumerate(blocks):
                    for pi in range(5):
                        for count,number in histogram.items():
                            rows.append((ci,bi,pi,0,0,count));mass.append(number)
            cohorts[label] = dict(sizes=np.array([1]),report=dict(joint_rng_blocks=[101,102]),
                count_histograms=dict(coordinates=np.array(rows),observations=np.array(mass),
                    shape=shape,windows_ms=np.array([5]),mass_checks=30,spike_checks=30))
        rows,distances,payload,report = analysis.compare_count_cdfs(cohorts,[5])
        np.testing.assert_array_equal(payload['count_support'],[0,2,4])
        expected = {'coarse':np.array([1/6,-1/6,0]), 'middle':np.array([1/12,-1/12,0])}
        for label,delta in expected.items():
            selected = [row for row in rows if row['left_cohort']==label
                        and row['condition']=='cue' and row['phase']=='late_delay']
            np.testing.assert_allclose([row['left_minus_right'] for row in selected],delta,atol=1e-12)
            maximum = [row['max_abs_empirical_cdf_difference'] for row in distances
                       if row['left_cohort']==label]
            np.testing.assert_allclose(maximum,np.max(np.abs(delta)),atol=1e-12)
        self.assertEqual(payload['coarse_cdf'].shape,(3,2,5,1,1,3))
        np.testing.assert_allclose(payload['coarse_cdf'][0,:,0,0,0],[[.5,.5,1],[0,1,1]])

    def test_declared_four_block_cohorts_keep_their_actual_denominators(self):
        roots=[self.root/('four_blocks_'+str(i)) for i in range(3)]
        for path,clock in zip(roots,analysis.CLOCKS):
            make_cohort(path,clock,seeds=[95001,95002,95003,95004])
        report=analysis.compare(roots,self.native,self.settings,self.root/'four_blocks')
        self.assertEqual(report['source_episodes_reused'],45)
        self.assertEqual(report['independent_joint_blocks_by_cohort'],
            {'numpy_dt_0p05':4,'numpy_dt_0p025':4,'numpy_dt_0p0125':4,'native_2560':3})
        self.assertIsNone(report['registered_cases_per_cohort'])
        self.assertEqual(report['registered_cases_by_cohort']['numpy_dt_0p05'],12)


if __name__ == '__main__':
    unittest.main()
