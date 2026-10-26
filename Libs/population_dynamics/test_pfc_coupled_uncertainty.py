"""Coupled whole-block resampling, frozen-domain binding and diagnostic limits."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import pfc_coupled_uncertainty as adapter
from . import pfc_numerical_error_budget
from .pfc_confirmation_uncertainty import (
    EMPIRICAL, binomial_probability_upper, bootstrap_primitives, construct_bands,
)
from .pfc_confirmation_covariance_scores import round_robin_pairings
from .pfc_numerical_precision import PrecisionSummary
from .provenance import sha256_file


def test_shared_resampling_design_uses_original_rng_algorithm():
    settings = dict(joint_rng_seeds=[95001, 95002, 95003, 95004], bootstrap_replicates=19, bootstrap_seed=97001)
    native = dict(calibration_precision=dict(status='COMPLETE_CALIBRATION_SAMPLING_PRECISION',
        settings=dict(settings, schema='pfc_numerical_precision_v1'),
        all_clocks_conditions_and_endpoints_share_resample_indices=True,
        implementation_sha256=sha256_file(Path(adapter.pfc_numerical_precision.__file__))))
    actual = adapter.shared_multiplicities(settings, native)
    rng = np.random.default_rng(97001)
    indices = rng.integers(0, 4, size=(19, 4))
    expected = np.asarray([np.bincount(row, minlength=4) for row in indices])
    np.testing.assert_array_equal(actual, expected)
    native['calibration_precision']['settings']['bootstrap_seed'] = 97002
    with pytest.raises(ValueError, match='BOUND_NATIVE_RESAMPLING_DESIGN_CHANGED'):
        adapter.shared_multiplicities(settings, native)


def test_all_linear_families_and_primitives_share_drawwise_maximum():
    m = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1], [2, 1, 0]])
    x = np.array([[0., 2., 1.], [3., 0., 1.], [.1, .1, .1]])
    summary = PrecisionSummary(len(m)); rows = []
    for index in range(len(x)):
        rows.extend(adapter.add_linear_precision(summary, x[index:index+1], x[index:index+1].mean(1),
                    [dict(family=['field_mean', 'count_cdf', 'source_response'][index])], m))
    draws = (m/3)@x[:2].T
    expected = (np.abs(draws-x[:2].mean(1))/draws.std(0, ddof=1)).max(1)
    np.testing.assert_allclose(summary.maximum, expected)
    assert rows[2]['bootstrap_standard_error'] == 0 and rows[2]['status'] == 'EMPIRICALLY_DEGENERATE'
    primitive = np.array([[2., 1., 3.]])
    _, _, maximum, _ = bootstrap_primitives(primitive, m, summary.maximum, 1)
    z = (m/3)@(primitive-primitive.mean(1)[:, None]).T
    np.testing.assert_allclose(maximum, np.maximum(expected, np.abs(z[:, 0])/z[:, 0].std(ddof=1)))


def explicit_lag_covariance(rates, indices, lag):
    x = rates[indices]
    a, b = (x, x) if lag == 0 else (x[:, :-lag], x[:, lag:])
    a = a-a.mean(0); b = b-b.mean(0)
    return np.einsum('bti,btj->ij', a, b)/((len(x)-1)*a.shape[1])


@pytest.mark.parametrize('blocks', [4, 32])
@pytest.mark.parametrize('lag', [0, 1])
def test_covariance_resamples_order_two_estimator_and_analytic_center(blocks, lag):
    rng = np.random.default_rng(601)
    left, right = rng.normal(size=(blocks, 5, 2)), rng.normal(size=(blocks, 5, 2))
    indices = rng.integers(0, blocks, size=(17, blocks))
    m = np.asarray([np.bincount(row, minlength=blocks) for row in indices])
    pairs, _ = round_robin_pairings(blocks); summary = PrecisionSummary(len(m))
    rows, pseudo, se, influence, residual = adapter.covariance_precision(
        left, right, lag, m, pairs, [dict(field=i) for i in range(4)], summary)
    point = (explicit_lag_covariance(left, np.arange(blocks), lag)
             -explicit_lag_covariance(right, np.arange(blocks), lag)).ravel()
    expected = np.asarray([(explicit_lag_covariance(left, row, lag)-explicit_lag_covariance(right, row, lag)).ravel()
                           for row in indices])
    np.testing.assert_allclose([row['estimate'] for row in rows], point, atol=1e-14)
    np.testing.assert_allclose([row['analytic_bootstrap_mean'] for row in rows], (blocks-1)/blocks*point, atol=1e-14)
    np.testing.assert_allclose([row['bootstrap_standard_error'] for row in rows], expected.std(0, ddof=1), atol=1e-14)
    np.testing.assert_allclose(summary.maximum, (np.abs(expected-(blocks-1)/blocks*point)/expected.std(0, ddof=1)).max(1), atol=1e-13)
    assert not np.allclose((m/blocks)@pseudo.T, expected)
    assert np.isfinite(se).all() and np.isfinite(influence).all() and np.isfinite(residual).all()


def test_first_order_covariance_degeneracy_does_not_remove_order_two_spread():
    left = np.repeat(np.array([0., 0., 2., 2.])[:, None, None], 3, axis=1)
    right = np.zeros_like(left)
    m = np.array([[4,0,0,0], [1,1,1,1], [0,0,4,0], [3,0,1,0], [1,0,3,0]])
    pairs, _ = round_robin_pairings(4); summary = PrecisionSummary(len(m))
    rows, pseudo, se, _, _ = adapter.covariance_precision(left, right, 0, m, pairs, [dict(field=0)], summary)
    np.testing.assert_allclose(pseudo, np.full_like(pseudo, 4/3), atol=1e-14)
    assert se[0] == 0 and rows[0]['bootstrap_standard_error'] > 0
    assert rows[0]['status'] == adapter.POSITIVE


def endpoint_fixture():
    ident = dict(left_cohort=adapter.LABELS[0], right_cohort=adapter.LABELS[1], condition='cue', phase='late_delay', field_index='0')
    identity = json.dumps(ident)
    domain = dict(family='field_mean', endpoint_identity=identity, definition_status=adapter.POSITIVE,
                  empirical_status=adapter.POSITIVE, physical_maximum_count='', estimate='.05')
    margin = dict(domain_row_index='0', family=domain['family'], endpoint_identity=identity,
                  definition_status=domain['definition_status'], margin='.2', units='Hz')
    stat = dict(margin, physical_maximum_count='', coupled_original_independent_blocks='3',
                coupled_signed_estimate='0', coupled_empirical_status=EMPIRICAL, coupled_score_kind='paired_block_linear_difference')
    return domain, margin, stat


def test_changed_empirical_zero_preserves_frozen_identity_and_estimate():
    domain, margin, stat = endpoint_fixture()
    current = adapter.bind_endpoints([domain], [margin], [stat], np.zeros((1, 3)), 3)[0]
    assert current['definition_status'] == adapter.POSITIVE and current['current_empirical_status'] == EMPIRICAL
    assert current['estimate'] == '0' and current['frozen_native_estimate'] == '.05'
    assert domain['estimate'] == '.05'


def test_new_positive_spread_does_not_relabel_frozen_empirical_definition():
    domain, margin, stat = endpoint_fixture()
    for row in (domain, margin, stat): row['definition_status'] = EMPIRICAL
    domain['estimate'] = '0'; stat['coupled_empirical_status'] = adapter.POSITIVE
    stat['coupled_signed_estimate'] = str(1/3)
    current = adapter.bind_endpoints([domain], [margin], [stat], np.array([[0., 1., 0.]]), 3)[0]
    assert current['definition_status'] == EMPIRICAL and current['current_empirical_status'] == adapter.POSITIVE
    assert current['frozen_native_estimate'] == '0'


@pytest.mark.parametrize('change', ['missing', 'index', 'identity', 'units', 'margin'])
def test_missing_reordered_or_changed_endpoint_inputs_are_rejected(change):
    domain, margin, stat = endpoint_fixture()
    if change == 'index': stat['domain_row_index'] = '1'
    if change == 'identity': stat['endpoint_identity'] = '{}'
    if change == 'units': stat['units'] = 'kHz'
    if change == 'margin': stat['margin'] = '.3'
    with pytest.raises(ValueError):
        adapter.bind_endpoints([domain], [margin], [] if change == 'missing' else [stat], np.zeros((1, 3)), 3)


def axes_fixture():
    seeds = [95001, 95002, 95003]
    fields = dict(source_rng_seeds=np.array(seeds), field_unit_counts=np.array(adapter.FIELD_SIZES),
        field_indices=np.arange(17), condition_names=np.array(adapter.CONDITIONS), time_start_ms=np.arange(1000)*5., bin_ms=np.array(5.), units=np.array('Hz'))
    responses = dict(source_rng_seeds=np.array(seeds), condition_names=np.array(adapter.CONDITIONS), phase_names=np.array(list(adapter.PHASES)),
        metric_names=np.array(adapter.METRICS), phase_start_ms=np.array([v[0] for v in adapter.PHASES.values()]),
        phase_stop_ms=np.array([v[1] for v in adapter.PHASES.values()]), units=np.array('Hz'))
    cdf = dict(source_rng_seeds=np.array(seeds), field_unit_counts=np.array(adapter.FIELD_SIZES), field_indices=np.arange(17),
        condition_names=np.array(adapter.CONDITIONS), phase_names=np.array(list(adapter.PHASES)), window_ms=np.array([5,50,100]),
        count_support=np.array([0,1]), units=np.array('probability'))
    for label in adapter.LABELS:
        for condition in adapter.CONDITIONS: fields[label+'_'+condition] = np.zeros((3,1000,17))
        responses[label+'_values'] = np.zeros((3,3,5,8))
        cdf[label+'_cdf'] = np.ones((3,3,5,3,17,2))
    scores = dict(source_rng_seeds=np.array(seeds), domain_row_indices=np.arange(2), scores=np.zeros((2,3)))
    return dict(fields=fields, responses=responses, cdf=cdf, scores=scores, seeds=seeds, rows=2)


def test_complete_explicit_axes_accepted():
    adapter.validate_axes(**axes_fixture())


@pytest.mark.parametrize('change', ['missing', 'blocks', 'conditions', 'fields', 'time', 'units', 'domain', 'shape'])
def test_invalid_array_axes_are_rejected(change):
    data = axes_fixture()
    if change == 'missing': del data['responses'][adapter.LABELS[0]+'_values']
    if change == 'blocks': data['cdf']['source_rng_seeds'] = data['cdf']['source_rng_seeds'][::-1]
    if change == 'conditions': data['responses']['condition_names'] = data['responses']['condition_names'][::-1]
    if change == 'fields': data['fields']['field_indices'] = np.arange(17)[::-1]
    if change == 'time': data['fields']['time_start_ms'][0] = 1.
    if change == 'units': data['responses']['units'] = np.array('kHz')
    if change == 'domain': data['scores']['domain_row_indices'] = np.array([1,0])
    if change == 'shape': data['fields'][adapter.LABELS[0]+'_'+adapter.CONDITIONS[0]] = np.zeros((3,17,1000))
    with pytest.raises(ValueError): adapter.validate_axes(**data)


def test_current_empirical_zero_retains_nonzero_joint_tail_uncertainty():
    ident = dict(left_cohort=adapter.LABELS[0], right_cohort=adapter.LABELS[1], condition='cue', phase='late_delay',
                 window_ms=5, field_index=0, count_threshold=0)
    domains = [dict(family='count_cdf', endpoint_identity=json.dumps(ident), definition_status=adapter.POSITIVE, estimate='0')]
    key = adapter.identity_key('count_cdf', ident)
    precision = {key: dict(status='EMPIRICALLY_DEGENERATE', estimate=0., bootstrap_standard_error=0.)}
    primitive = (np.ones(2), np.zeros(2), np.zeros(2), np.ones(2))
    lookup = {('cdf', label, 'cue', 'late_delay', 5., 0, 0): i for i, label in enumerate(adapter.LABELS[:2])}
    tail = binomial_probability_upper(0, 128, .025)
    bands, _, _ = construct_bands(domains, precision, primitive, lookup, [0], {(5.,0):3}, 4., tail)
    np.testing.assert_allclose([bands[0]['lower'], bands[0]['upper']], [-tail, tail])
    assert bands[0]['definition_status'] == adapter.POSITIVE and tail > .01


def test_frozen_budget_consumed_without_rebuilding_margins(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('frozen margin rebuild is forbidden')
    monkeypatch.setattr(pfc_numerical_error_budget, 'build_margins', forbidden)
    domain, margin, _ = endpoint_fixture(); original = copy.deepcopy(margin)
    band = dict(domain_row_index=0, family=domain['family'], endpoint_identity=domain['endpoint_identity'],
                definition_status=domain['definition_status'], estimate=.04, lower=-.1, upper=.15,
                planning_lower_root_n_and_zero_tail_at_128=-.05, planning_upper_root_n_and_zero_tail_at_128=.1)
    rows, _ = adapter.frozen_budget_assessment([margin], [band], 128)
    assert rows[0]['planning_status'] == 'PLANNING_BAND_WITHIN_BUDGET'
    assert rows[0]['margin'] == '.2' and margin == original


def test_real_cli_rejects_confirmation_scope_before_outputs(tmp_path):
    config = tmp_path/'wrong.yaml'; config.write_text('schema: pfc_coupled_uncertainty_diagnostic_v1\nscope: confirmation\n')
    result = subprocess.run([sys.executable, '-B', '-m', 'Libs.population_dynamics.pfc_coupled_uncertainty',
                             '--config', str(config), '--project-root', str(tmp_path), '--output', str(tmp_path/'out')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'CALIBRATION_DIAGNOSTIC_ONLY_REQUIRED' in result.stderr
    assert not (tmp_path/'out').exists()


@pytest.fixture
def conditional_history_fixture():
    # Only registered count identities need values for this pure preparation.
    domains = [dict(family='field_mean', definition_status=adapter.POSITIVE) for _ in range(84330)]
    for row in domains[-1734:]:
        row['definition_status'] = adapter.UNDEFINED
    profiles = []
    for left, right in zip(adapter.LABELS[:-1], adapter.LABELS[1:]):
        for phase in ('cue', 'early_delay', 'middle_delay', 'late_delay'):
            for field in ((4, 5, 6) if phase == 'cue' else (3, 4, 5, 6, 7)):
                i = len(profiles)
                profile = dict(left_cohort=left, right_cohort=right, phase=phase,
                               field_index=field, condition='strong_cue', window_ms=5.)
                profile['endpoints'] = [dict(row_index=10+2*i, count_threshold=1)]
                if i % 3:
                    profile['endpoints'].append(dict(row_index=11+2*i, count_threshold=2))
                for endpoint in profile['endpoints']:
                    identity = {name: str(profile[name]) if name in ('field_index', 'window_ms')
                                else profile[name] for name in adapter.IDENTITY['count_cdf']
                                if name != 'count_threshold'}
                    identity['count_threshold'] = str(endpoint['count_threshold'])
                    domains[endpoint['row_index']] = dict(family='count_cdf',
                        endpoint_identity=json.dumps(identity), definition_status=adapter.POSITIVE,
                        units='probability')
                profiles.append(profile)
    settings = dict(required_source_seeds=list(adapter.SEEDS), profiles=profiles,
                    auxiliary_repeats=['A', 'B'])
    records = []
    def stats(mean):
        return dict(sample_mean=float(mean), within_archive_mean_variance_diagnostic=2.,
                    inner_mc_mean_variance_diagnostic=2.25,
                    unit_window_sampling_mean_variance_diagnostic=-.25)
    for block, seed in enumerate(adapter.SEEDS):
        for repeat in ('A', 'B'):
            for i, profile in enumerate(profiles):
                common = .02*block + .1*np.sin(block+i)
                value = common if repeat == 'A' else -.01*block+.07*np.cos(block+2*i)
                paired = stats(value)
                paired['fine_minus_coarse_cdf_le1'] = -paired['sample_mean']
                records.append(dict(source_rng_seed=seed, key=adapter.profile_key(profile, repeat),
                    coarse=stats(.25), fine=stats(.25+value), paired=paired,
                    provenance=dict(kind='synthetic_test', source=seed)))
    return settings, records, domains


def test_conditional_history_complete_mapping_sign_and_fixed_midpoint(conditional_history_fixture):
    settings, records, domains = conditional_history_fixture
    original_domains = copy.deepcopy(domains)
    prepared = adapter.conditional_history_scores(settings, reversed(records), domains, adapter.SEEDS)
    assert prepared['coverage']['completed_pairs'] == 2304
    assert prepared['coverage']['full_registered_scope_covered'] is True
    assert prepared['scores'].shape == (108, 32)
    np.testing.assert_array_equal(prepared['source_rng_seeds'], adapter.SEEDS)
    indexed = {(record['source_rng_seed'], record['key']): record for record in records}
    for coordinate, identity in enumerate(prepared['identities']):
        i = coordinate % 36
        assert identity['domain_row_index'] == settings['profiles'][i]['endpoints'][0]['row_index']
        assert identity['count_threshold'] == 1 and identity['units'] == 'probability'
        if coordinate < 72:
            key = identity['profile_key']
            expected = [indexed[seed, key]['paired']['sample_mean'] for seed in adapter.SEEDS]
            np.testing.assert_array_equal(prepared['scores'][coordinate], expected)
            np.testing.assert_array_equal(prepared['scores'][coordinate],
                [-indexed[seed, key]['paired']['fine_minus_coarse_cdf_le1'] for seed in adapter.SEEDS])
        else:
            assert identity['merger_weights'] == [.5, .5]
            assert identity['source_profile_keys'] == [adapter.profile_key(settings['profiles'][i], repeat)
                                                       for repeat in ('A', 'B')]
    np.testing.assert_array_equal(prepared['scores'][72:], .5*(prepared['scores'][:36]+prepared['scores'][36:72]))
    assert len(prepared['unresolved_threshold2']) == 24
    assert all(row['conditional_history_score_computed'] is False for row in prepared['unresolved_threshold2'])
    assert prepared['coverage']['records'][0]['paired']['unit_window_sampling_mean_variance_diagnostic'] == -.25
    assert prepared['negative_variance_diagnostics_preserved'] is True
    assert prepared['variance_noise_subtraction_applied_to_bootstrap_scores'] is False
    assert domains == original_domains


@pytest.mark.parametrize('change', ['missing_pair', 'duplicate_pair', 'source_axis', 'threshold1_missing',
                                    'mapped_identity', 'mapped_units', 'undefined', 'structural',
                                    'duplicate_domain_identity', 'full_domain'])
def test_conditional_history_rejects_incomplete_or_changed_scope(conditional_history_fixture, change):
    settings, records, domains = conditional_history_fixture
    source_seeds = adapter.SEEDS
    row_index = settings['profiles'][0]['endpoints'][0]['row_index']
    if change == 'missing_pair': records.pop()
    if change == 'duplicate_pair': records.append(records[0])
    if change == 'source_axis': source_seeds = source_seeds[::-1]
    if change == 'threshold1_missing': settings['profiles'][0]['endpoints'] = []
    if change == 'mapped_identity':
        identity = json.loads(domains[row_index]['endpoint_identity'])
        identity['field_index'] = 9
        domains[row_index]['endpoint_identity'] = json.dumps(identity)
    if change == 'mapped_units': domains[row_index]['units'] = 'Hz'
    if change == 'undefined':
        domains[row_index]['definition_status'] = adapter.UNDEFINED
        domains[-1]['definition_status'] = adapter.POSITIVE
    if change == 'structural': domains[row_index]['definition_status'] = adapter.STRUCTURAL
    if change == 'duplicate_domain_identity': domains[100] = dict(domains[row_index])
    if change == 'full_domain': domains.pop()
    with pytest.raises(ValueError):
        adapter.conditional_history_scores(settings, records, domains, source_seeds)


def test_conditional_history_augments_one_joint_maximum_with_shared_source_draws(conditional_history_fixture):
    settings, records, domains = conditional_history_fixture
    prepared = adapter.conditional_history_scores(settings, records, domains, adapter.SEEDS)
    multiplicities = adapter.paired_bootstrap_multiplicities(32, 5000, 97001)
    original_scores = np.vstack((np.arange(32)/32, np.sin(np.arange(32))))
    original_summary = PrecisionSummary(5000)
    adapter.add_linear_precision(original_summary, original_scores, original_scores.mean(1),
                                 [dict(source='original_1'), dict(source='original_2')], multiplicities)
    original_maximum = original_summary.maximum.copy()
    result = adapter.augment_conditional_history_precision(original_maximum, multiplicities, prepared)
    all_scores = np.concatenate((original_scores, prepared['scores']))
    draws = (multiplicities/32)@all_scores.T
    expected = (np.abs(draws-all_scores.mean(1))/draws.std(0, ddof=1)).max(1)
    np.testing.assert_allclose(result['joint_maximum'], expected, atol=1e-13)
    np.testing.assert_array_equal(original_maximum, original_summary.maximum)
    for i in range(36):
        np.testing.assert_allclose(draws[:, 2+72+i], .5*(draws[:, 2+i]+draws[:, 2+36+i]), atol=1e-14)
    assert len(result['conditional_history_precision']) == 108
    assert result['diagnostics']['original_domain_rows'] == 84330
    assert result['diagnostics']['original_undefined_domain_rows'] == 1734
    assert result['diagnostics']['conditional_history_precision']['requested_endpoints'] == 108
    assert result['diagnostics']['bands_computed'] is False
    assert result['diagnostics']['maximum_is_simultaneous_confidence_statement'] is False
    assert 'conditional_history_maximum' not in result


@pytest.mark.parametrize('change', ['multiplicities', 'draw_order', 'maximum', 'midpoint', 'point'])
def test_conditional_history_augmentation_rejects_changed_original_draws(conditional_history_fixture, change):
    settings, records, domains = conditional_history_fixture
    prepared = adapter.conditional_history_scores(settings, records, domains, adapter.SEEDS)
    multiplicities = adapter.paired_bootstrap_multiplicities(32, 5000, 97001)
    maximum = np.ones(5000)
    if change == 'multiplicities': multiplicities = adapter.paired_bootstrap_multiplicities(32, 5000, 97002)
    if change == 'draw_order': multiplicities = multiplicities[::-1]
    if change == 'maximum': maximum = maximum[:-1]
    if change == 'midpoint': prepared['scores'][72, 0] += .001
    if change == 'point': prepared['point'][0] += .001
    with pytest.raises(ValueError):
        adapter.augment_conditional_history_precision(maximum, multiplicities, prepared)


def test_conditional_history_real_cli_complete_and_missing_coverage(conditional_history_fixture, tmp_path):
    import hashlib
    import yaml
    settings0, records, domains = conditional_history_fixture
    config = dict(schema=adapter.SCHEMA, scope='development_calibration',
        method_adopted_for_confirmation=False, confirmation_execution_released=False,
        coverage_qualified=False, cohorts=list(adapter.LABELS), joint_rng_seeds=list(adapter.SEEDS),
        bootstrap_replicates=5000, bootstrap_seed=97001, planning_blocks=128,
        nominal_alpha=.05, joint_tail_alpha=.025, endpoint_chunk_size=256,
        single_block_influence_fraction_flag=.25,
        inputs={key:dict(path=key, report_sha256='pending')
                for key in ('coupled', 'domain', 'budget', 'linear', 'resampling_design')})
    def artifact(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value).encode()
        path.write_bytes(payload)
        return dict(path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    def table(path, rows):
        adapter.write_rows(path, rows)
        return dict(bytes=path.stat().st_size, sha256=sha256_file(path))
    for index, domain in enumerate(domains):
        domain.setdefault('endpoint_identity', json.dumps(dict(synthetic_fixture_row=index)))
        domain.setdefault('units', 'Hz')
    domain_root, budget_root, donor_root = (tmp_path/name for name in ('domain', 'budget', 'donor'))
    for path in (domain_root, budget_root, donor_root): path.mkdir()
    domain_output = table(domain_root/'endpoint_domains.csv', domains)
    margins = [dict(domain_row_index=i, family=row['family'], endpoint_identity=row['endpoint_identity'],
                    definition_status=row['definition_status'], units=row.get('units', 'Hz'))
               for i, row in enumerate(domains)]
    margin_output = table(budget_root/'frozen_numerical_margins.csv', margins)
    config['inputs']['domain']['report_sha256'] = artifact(domain_root/'report.json',
        dict(outputs={'endpoint_domains.csv':domain_output}))['sha256']
    config['inputs']['budget']['report_sha256'] = artifact(budget_root/'report.json',
        dict(outputs={'frozen_numerical_margins.csv':margin_output}))['sha256']
    multiplicities = adapter.paired_bootstrap_multiplicities(32, 5000, 97001)
    np.savez_compressed(donor_root/'joint_bootstrap_maximum.npz', direct_maximum=np.ones(5000),
        joint_maximum=np.ones(5000), bootstrap_multiplicities=multiplicities, source_rng_seeds=adapter.SEEDS)
    donor = dict(status='COMPLETE_COUPLED_CALIBRATION_JOINT_UNCERTAINTY_DIAGNOSTIC',
        domain_rows=84330, undefined_endpoints=1734, original_independent_blocks=32,
        all_endpoints_share_identical_original_block_multiplicities=True, coverage_qualified=False,
        settings=copy.deepcopy(config), frozen_margin_sha256=margin_output['sha256'], joint_critical_value=1.,
        outputs={'joint_bootstrap_maximum.npz':dict(bytes=(donor_root/'joint_bootstrap_maximum.npz').stat().st_size,
                    sha256=sha256_file(donor_root/'joint_bootstrap_maximum.npz'))})
    donor_sha = artifact(donor_root/'report.json', donor)['sha256']
    coverage_root = tmp_path/'coverage'
    history_path = coverage_root/'coverage_inputs/foundation/source_config.yaml'
    history_path.parent.mkdir(parents=True)
    history_path.write_text(yaml.safe_dump(settings0))
    coverage = adapter.pfc_profile_history_coverage.aggregate(settings0, list(adapter.SEEDS), records)
    report_entry = artifact(coverage_root/'analysis/report.json', coverage)
    report_entry['path'] = 'analysis/report.json'
    history_entry = dict(path='coverage_inputs/foundation/source_config.yaml', bytes=history_path.stat().st_size,
                         sha256=sha256_file(history_path))
    artifact(coverage_root/'delivery.json', dict(verified=True,
        status='COMPLETE_PFC_PROFILE_HISTORY_COVERAGE_COMPLETE', files=[report_entry, history_entry]))
    config['conditional_history'] = dict(coverage_archive='coverage', history_config_sha256=history_entry['sha256'],
        joint_precision=dict(path='donor', report_sha256=donor_sha), midpoint_weights=[.5, .5])
    config_path = tmp_path/'history.yaml'
    config_path.write_text(yaml.safe_dump(config))
    def cli(output):
        return subprocess.run([sys.executable, '-B', '-m', 'Libs.population_dynamics.pfc_coupled_uncertainty',
            '--config', str(config_path), '--project-root', str(tmp_path), '--output', str(output)],
            capture_output=True, text=True)
    output = tmp_path/'complete'
    result = cli(output)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'COMPLETE_CONDITIONAL_HISTORY_JOINT_EMPIRICAL_PRECISION'
    assert report['registered_pairs'] == 2304 and report['original_domain_rows'] == 84330
    assert report['bands_computed'] is False and report['production_episodes'] == 0
    assert len(adapter.read_rows(output/'conditional_history_sampling_precision.csv')) == 108
    with np.load(output/'conditional_history_joint_precision.npz') as bank:
        np.testing.assert_array_equal(bank['bootstrap_multiplicities'], multiplicities)
        assert bank['conditional_history_scores'].shape == (108, 32)
    coverage['full_registered_scope_covered'] = False
    coverage['completed_pairs'] = 72
    report_entry = artifact(coverage_root/'analysis/report.json', coverage)
    report_entry['path'] = 'analysis/report.json'
    artifact(coverage_root/'delivery.json', dict(verified=True,
        status='COMPLETE_PFC_PROFILE_HISTORY_COVERAGE_COMPLETE', files=[report_entry, history_entry]))
    failed_output = tmp_path/'incomplete'
    failure = cli(failed_output)
    assert failure.returncode != 0 and 'COMPLETE2304_DELIVERY_REQUIRED' in failure.stderr
    assert not failed_output.exists()
