"""Direct unit-matrix reference for the saved-ensemble Gram calculation."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from . import unit_covariance_analysis as analysis


def test_gram_matches_direct_off_diagonal_outer_products():
    rng = np.random.default_rng(71)
    a, b = rng.normal(size=(8, 7)), rng.normal(size=(8, 7))
    mask = ~np.eye(7, dtype=bool)
    direct = np.array([[np.mean((np.outer(x, x) * np.outer(y, y))[mask]) for y in b] for x in a])
    for chunk in (1, 3, 20):
        np.testing.assert_allclose(analysis.off_diagonal_kernel(a, b, chunk_units=chunk), direct, atol=1e-14)


def test_summary_matches_direct_covariance_and_delete_pair():
    rng = np.random.default_rng(42)
    native, learned = rng.poisson(3, (16, 9)), rng.poisson(5, (16, 9))
    result, sensitivity, kernels = analysis.describe_totals(native, learned, chunk_units=4)
    a, b = analysis.paired_differences(native), analysis.paired_differences(learned)
    ca, cb = a.T @ a / len(a), b.T @ b / len(b)
    mask = ~np.eye(9, dtype=bool)
    np.testing.assert_allclose(result['empirical_difference'], ((ca - cb) ** 2)[mask].mean(), atol=1e-13)
    np.testing.assert_allclose(result['corrected_difference'], result['empirical_difference'] - result['sampling_noise'], atol=1e-13)
    assert result['native_sampling_noise'] >= 0 and result['learned_sampling_noise'] >= 0
    assert len(sensitivity) == 16
    expected = analysis.covariance_summary(kernels['native'][1:, 1:], kernels['learned'], kernels['cross'][1:])
    assert sensitivity[0]['corrected_difference'] == expected['corrected_difference']
    assert result['delete_one_pair_minimum'] == min(r['corrected_difference'] for r in sensitivity)


def test_signed_sampling_correction_is_preserved():
    y = np.array([[1., 1.], [1., -1.], [1., 1.], [1., -1.]])
    native = np.zeros((8, 2)); native[::2] = np.sqrt(2.) * y
    result, _, _ = analysis.describe_totals(native, np.zeros_like(native))
    assert result['corrected_difference'] == pytest.approx(-1 / 3)
    assert result['empirical_difference'] == pytest.approx(0, abs=1e-15)
    assert result['sampling_noise'] == pytest.approx(1 / 3)


def test_horizon_totals_use_each_unit_and_complete_time_prefix():
    counts = np.arange(8 * 5 * 11).reshape(8, 5, 11)
    totals = analysis.horizon_totals(counts, (1, 3, 5), chunk_units=3)
    for h in totals:
        np.testing.assert_array_equal(totals[h], counts[:, :h].sum(axis=1))
    counts[0, -1, -1] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        analysis.horizon_totals(counts, (1,))
    with pytest.raises(ValueError, match='integer'):
        analysis.horizon_totals(counts.astype(float), (1,))


@pytest.mark.parametrize('counts,bins,chunk', [(np.zeros((8, 2, 3), dtype=int), (3,), 2),
                                              (np.zeros((8, 2, 3), dtype=int), (1,), 0),
                                              (np.full((8, 2, 3), 2**53, dtype=np.uint64), (1,), 2)])
def test_invalid_count_domains_fail(counts, bins, chunk):
    with pytest.raises(ValueError):
        analysis.horizon_totals(counts, bins, chunk_units=chunk)


def alignment_fixture():
    binding = dict(episode_report_sha256='episode', native_source_identity={'dt_ms': .1})
    nr = dict(status='DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT', draws=16, boundary_tick=10000,
              generated_bins=100, reference_unit_count=100000, field_count=128,
              reference_binding={**binding, 'fitting_scale_sha256': 'scale'},
              reference_source_identity=binding['native_source_identity'], scientific_claim_qualified=False)
    lr = dict(status='EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT', draws=16, boundary_tick=10000,
              generated_bins=100, unit_count=100000, field_count=128, training_seed=41,
              source_binding=binding, native_source_identity=binding['native_source_identity'],
              fitting_scale_sha256='scale', scientific_claim_qualified=False,
              configuration=dict(case_id='train_block_014_familiar_0', bin_ms=5., native_dt_ms=.1,
                                 boundary_time_ms=1000., horizon_ms=500., formal_evaluation=False))
    mapping = np.arange(100000) % 128
    nf = dict(unit_ids=np.arange(100000), unit_field_index=mapping,
              field_ids=np.array([str(i) for i in range(128)]), unit_count=np.bincount(mapping),
              observed_unit_counts=np.broadcast_to(np.zeros((1, 100000), dtype=int), (100, 100000)),
              observed_field_counts=np.zeros((100, 128), dtype=int), stimulus=np.zeros((100, 8)))
    lf = {**nf, 'stimulus': nf['stimulus'][None]}
    return nr, lr, nf, lf


def test_source_units_observation_and_stimulus_alignment():
    nr, lr, nf, lf = alignment_fixture()
    analysis.validate_alignment(nr, lr, nf, lf)
    changed = {**lf, 'unit_ids': lf['unit_ids'][::-1]}
    with pytest.raises(ValueError, match='identity differs'):
        analysis.validate_alignment(nr, lr, nf, changed)
    changed = {**lf, 'stimulus': lf['stimulus'] + 1}
    with pytest.raises(ValueError, match='stimulus differs'):
        analysis.validate_alignment(nr, lr, nf, changed)
    changed = copy.deepcopy(lr); changed['source_binding']['episode_report_sha256'] = 'other'
    with pytest.raises(ValueError, match='binding'):
        analysis.validate_alignment(nr, changed, nf, lf)


def test_delivery_staging_rejects_corrupted_or_unreplayed_inputs(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
    import run_unit_covariance_analysis as runner
    root = tmp_path / 'producer'; (root / 'analysis').mkdir(parents=True)
    for name in ('sampled_unit_counts.npy', 'ensemble_fields.npz'):
        (root / 'analysis' / name).write_bytes(b'tiny staging fixture')
    report = dict(artifacts={name: runner.digest(root / 'analysis' / name)
                            for name in ('sampled_unit_counts.npy', 'ensemble_fields.npz')})
    (root / 'analysis/analysis.json').write_text(json.dumps(report))
    rows = [dict(path='analysis/' + name, bytes=(root / 'analysis' / name).stat().st_size,
                 sha256=runner.digest(root / 'analysis' / name)) for name in
            ('analysis.json', 'sampled_unit_counts.npy', 'ensemble_fields.npz')]
    (root / 'delivery.json').write_text(json.dumps(dict(status='PASS', exit_code=0,
                                                      scientific_claim_qualified=False, files=rows)))
    replay = tmp_path / 'replay.json'
    accepted = dict(status='PASS', scientific_claim_qualified=False, exact_unit_count_values=160000000,
                    delivery_sha256=runner.digest(root / 'delivery.json'),
                    analysis_sha256=runner.digest(root / 'analysis/analysis.json'))
    replay.write_text(json.dumps(accepted))
    result = runner.stage_ensemble(root, replay, tmp_path / 'work', native=True)
    assert result['delivery_sha256'] == accepted['delivery_sha256']
    (root / 'analysis/sampled_unit_counts.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='manifest identity changed'):
        runner.stage_ensemble(root, replay, tmp_path / 'corrupt', native=True)
    replay.write_text(json.dumps({**accepted, 'status': 'FAILED'}))
    with pytest.raises(ValueError, match='independently replayed'):
        runner.stage_ensemble(root, replay, tmp_path / 'unreplayed', native=True)


def test_empirical_energy_matches_direct_sample_covariance():
    totals = np.random.default_rng(48).poisson(3, (16, 7))
    covariance = np.cov(totals, rowvar=False, ddof=1)
    mask = ~np.eye(7, dtype=bool)
    assert analysis.empirical_covariance_energy(totals) == pytest.approx((covariance**2)[mask].mean())


def test_shuffle_preserves_marginals_and_horizon_alignment():
    totals = np.arange(16 * 13).reshape(16, 13)
    shuffled = analysis.shuffle_unit_draws(totals)
    np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(totals, axis=0))
    np.testing.assert_array_equal(analysis.shuffle_unit_draws(totals), shuffled)
    np.testing.assert_array_equal(analysis.shuffle_unit_draws(totals * 3), shuffled * 3)
    assert not np.array_equal(shuffled, totals)
    # At least two units must have different permutations.
    assert not np.array_equal(shuffled[:, 0] // 13, shuffled[:, 1] // 13)


def test_retained_cell_staging_binds_delivery_and_preserves_source(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
    import run_unit_covariance_analysis as runner
    root = tmp_path / 'cell'
    rows = []
    for name in ('training/run.json', 'forecasts/evaluation.json', 'forecasts/full_unit_samples.npz', 'replay.json'):
        source = root / name; source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(b'fixture')
        rows.append(dict(path=name, bytes=source.stat().st_size, sha256=runner.digest(source)))
    proof = dict(status='PASS', exit_code=0, training_seed=41, scientific_claim_qualified=False,
                 calibration_or_test_accessed=False, files=rows)
    (root / 'delivery.json').write_text(json.dumps(proof))
    result = runner.stage_retained_cell(root, 41, tmp_path / 'work')
    assert result['sample_sha256'] == rows[2]['sha256']
    with pytest.raises(ValueError, match='development-only'):
        runner.stage_retained_cell(root, 42, tmp_path / 'wrong_seed')
    (root / 'forecasts/full_unit_samples.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='manifest identity changed'):
        runner.stage_retained_cell(root, 41, tmp_path / 'corrupt')


def test_completed_analysis_reuse_reads_tables_without_counts(tmp_path):
    # This is a delivered description fixture. Its original sample path does
    # not exist: reuse must consume only the bound small analysis artifacts.
    root = tmp_path / 'previous'; root.mkdir()
    row = dict(representation='recurrent', seed=41, horizon_ms=20, corrected_energy=-.1)
    analysis._write_csv(root / 'retained_unit_covariance.csv', [row] * 3)
    sensitivity = [dict(deleted_pair=i, corrected_energy=-.1) for i in range(8)] * 3
    analysis._write_csv(root / 'delete_pair_sensitivity.csv', sensitivity)
    np.savez(root / 'covariance_kernels.npz', recurrent_41_bins4=np.zeros((8, 8)))
    report = dict(status='DESCRIBED_RETAINED_POPULATION_COVARIANCE_DEVELOPMENT',
                  scientific_claim_qualified=False, formal_test_accessed=False,
                  case_id='train_block_014_familiar_0', bin_ms=5., horizons_ms=[20, 80, 500],
                  permutation_seed=analysis.PERMUTATION_SEED,
                  draw_pairs=[[i, i + 1] for i in range(0, 16, 2)], source_identity={'cohort': 'fixture'},
                  summary=[row] * 3, cells=[dict(representation='recurrent', seed=41, input='/missing/counts')],
                  artifacts={name: analysis.sha256_file(root / name) for name in
                             ('retained_unit_covariance.csv', 'delete_pair_sensitivity.csv', 'covariance_kernels.npz')})
    (root / 'analysis.json').write_text(json.dumps(report))
    result = analysis.analyze_retained_cells([], tmp_path / 'combined', previous_analyses=[root])
    assert result['summary'] == report['summary']
    assert result['cells'] == report['cells']
    with pytest.raises(ValueError, match='duplicate'):
        analysis.analyze_retained_cells([], tmp_path / 'duplicate', previous_analyses=[root, root])
    (root / 'retained_unit_covariance.csv').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        analysis.reuse_retained_analysis(root)
