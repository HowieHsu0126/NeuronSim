"""Small synthetic contract fixtures; these tests never simulate PFC episodes."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from . import pfc_temporal_control_comparison as temporal

SHA = 'a'*64


def settings():
    return dict(schema=temporal.SCHEMA, scope=temporal.SCOPE,
        protocol=dict(path='protocol.yaml', sha256=SHA),
        original_source=dict(archive='original', delivery_sha256=SHA),
        original_comparison=dict(archive='comparison', delivery_sha256=SHA, report_sha256=SHA),
        source_archive='reversed', source_job_id=601654,
        confirmation_execution_released=False, scientific_fidelity_qualified=False, production_episodes=0)


@pytest.mark.parametrize('key,value', [
    ('schema', 'pfc_coupled_clock_comparison_v1'), ('scope', 'confirmation'),
    ('confirmation_execution_released', True), ('scientific_fidelity_qualified', True),
    ('production_episodes', 1), ('production_episodes', False),
])
def test_refuses_scope_promotion(key, value):
    config = settings()
    config[key] = value
    with pytest.raises(ValueError, match='DESCRIPTIVE_SCOPE'):
        temporal.validate_settings(config)


@pytest.mark.parametrize('key', ['confidence_intervals', 'covariance_estimates', 'fdr_tests', 'frozen_margin_tests'])
def test_single_reused_block_cannot_request_inference(key):
    config = settings()
    config[key] = True
    with pytest.raises(ValueError, match='NO_INFERENTIAL_QUALIFICATION'):
        temporal.validate_settings(config)


def test_valid_registration_and_cli_refusal_precede_analysis_outputs(tmp_path):
    temporal.validate_settings(settings())
    config = settings()
    config['scope'] = 'confirmation'
    path = tmp_path/'settings.yaml'
    path.write_text(yaml.safe_dump(config))
    output = tmp_path/'analysis'
    with pytest.raises(ValueError, match='DESCRIPTIVE_SCOPE'):
        temporal.run(path, tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize('change', ['unfinished', 'unverified', 'wrong_job'])
def test_archive_requires_actual_complete_bound_delivery(tmp_path, change):
    delivery = dict(status=temporal.SOURCE_STATUS, verified=True, job_id='601654', files=[])
    if change == 'unfinished':
        delivery['status'] = 'RUNNING_COUPLED_CLOCK_TEMPORAL_CONTROL'
    elif change == 'unverified':
        delivery['verified'] = False
    else:
        delivery['job_id'] = '591779'
    (tmp_path/'delivery.json').write_text(json.dumps(delivery))
    with pytest.raises(ValueError, match='COMPLETE_VERIFIED_TEMPORAL_INPUT'):
        temporal.read_archive(tmp_path, temporal.SOURCE_STATUS, job_id=601654)


def original_comparison():
    reference = settings()['original_source']
    report = dict(status=temporal.ORIGINAL_COMPARISON_STATUS, selected_source_rng_seed=temporal.SEED,
        original_independent_blocks=1, pilot_delivery_sha256=SHA,
        settings=dict(pilot_archive=reference['archive']),
        pilot_cases=[dict(dt_ms=dt, case_id=f'pfc_{temporal.SEED}_{condition}', report_sha256=SHA)
                     for _, dt in temporal.CLOCKS for condition in temporal.CONDITIONS])
    return report, reference


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'seed', 'delivery'])
def test_original_primitives_require_all_accepted_source_case_bindings(change):
    report, reference = original_comparison()
    assert len(temporal.accepted_original_cases(report, reference)) == 6
    if change == 'missing':
        report['pilot_cases'].pop()
    elif change == 'duplicate':
        report['pilot_cases'][-1] = dict(report['pilot_cases'][0])
    elif change == 'seed':
        report['pilot_cases'][0]['case_id'] = 'pfc_95002_omission'
    else:
        report['pilot_delivery_sha256'] = 'b'*64
    with pytest.raises(ValueError):
        temporal.accepted_original_cases(report, reference)


def temporal_case(dt=.025, condition='cue'):
    modules = (temporal.execution.executor, temporal.execution.native_drive,
               temporal.execution.native_execution, temporal.execution.pfc_clock_coupling)
    code = {module.__name__.split('.')[-1]: SHA for module in modules}
    original = dict(status='COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE', source_execution_complete=True,
        source_rng_seed=temporal.SEED, refinement_seeds=[95411, 95412],
        case_id=f'pfc_{temporal.SEED}_{condition}', condition=condition, dt_ms=dt, coarse_dt_ms=.05,
        coupled_core_units=100000, duration_ms=5000.,
        resolved_config=dict(dt_ms=dt, duration_ms=5000., readout_bin_ms=5., excitatory_neurons=80000,
                             inhibitory_neurons=20000, excitatory_preference_bins=16),
        selected_case=dict(case_id=f'pfc_{temporal.SEED}_{condition}', source_rng_seed=temporal.SEED),
        coarse_reference=dict(archive='coarse', delivery_sha256=SHA), code=code,
        inputs_used=dict(input_mode='coupled_binomial_clock_refinement_stream',
                         E=dict(events=123, values=1000), I=dict(events=456, values=500)),
        drive=dict(coupling={}, external_event_totals={'E': 123, 'I': 456}))
    protocol = dict(original_fine_reference=settings()['original_source'],
                    coarse_references={condition: original['coarse_reference']})
    report = copy.deepcopy(original)
    control = dict(transform=temporal.SCHEME)
    report.update(status='COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE',
        scope=temporal.execution.TEMPORAL_CONTROL_SCOPE, settings_sha256=SHA,
        original_fine_reference=protocol['original_fine_reference'], original_fine_report_sha256=SHA,
        temporal_transform=temporal.SCHEME, new_original_independent_blocks=0,
        reused_original_independent_blocks=1, new_auxiliary_streams=0, rng_streams_replayed=True,
        independent_auxiliary_repeat=False, confirmation_execution_released=False,
        scientific_fidelity_qualified=False, production_episodes=0,
        original_fine_input_agreement={key: True for key in temporal.INPUT_CHECKS},
        coarse_final_rng_matches_archived_episode=True, coarse_current_agreement={'current_A': True},
        validations={'complete_source': {'passed': True}})
    report['inputs_used']['input_order_control'] = control
    report['drive']['coupling']['input_order_control'] = control
    report['code']['pfc_clock_temporal_control'] = 'b'*64
    return original, protocol, report


def test_exact_replay_identity_accepts_only_declared_order_transform():
    original, protocol, report = temporal_case()
    temporal.validate_temporal_case(report, original, protocol, SHA, SHA, .025, 'cue')


@pytest.mark.parametrize('change', [
    'unfinished', 'dt', 'condition', 'seed', 'refinement', 'protocol', 'original_report',
    'original_reference', 'config', 'coarse_reference', 'new_block', 'independent_repeat',
    'false_input_receipt', 'missing_input_receipt', 'source_failure', 'input_events',
    'extra_input_change', 'unbound_control', 'event_totals', 'numerical_kernel',
])
def test_temporal_case_refuses_identity_input_and_incomplete_source_drift(change):
    original, protocol, report = temporal_case()
    if change == 'unfinished':
        report['status'] = 'RUNNING_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE'
    elif change == 'dt':
        report['dt_ms'] = .0125
    elif change == 'condition':
        report['condition'] = 'omission'
    elif change == 'seed':
        report['source_rng_seed'] = 95002
    elif change == 'refinement':
        report['refinement_seeds'] = [95413, 95414]
    elif change == 'protocol':
        report['settings_sha256'] = 'b'*64
    elif change == 'original_report':
        report['original_fine_report_sha256'] = 'b'*64
    elif change == 'original_reference':
        report['original_fine_reference'] = dict(archive='other', delivery_sha256=SHA)
    elif change == 'config':
        report['resolved_config']['readout_bin_ms'] = 10.
    elif change == 'coarse_reference':
        report['coarse_reference'] = dict(archive='other', delivery_sha256=SHA)
    elif change == 'new_block':
        report['new_original_independent_blocks'] = 1
    elif change == 'independent_repeat':
        report['independent_auxiliary_repeat'] = True
    elif change == 'false_input_receipt':
        report['original_fine_input_agreement'][temporal.INPUT_CHECKS[0]] = False
    elif change == 'missing_input_receipt':
        report['original_fine_input_agreement'].pop(temporal.INPUT_CHECKS[1])
    elif change == 'source_failure':
        report['validations']['complete_source']['passed'] = False
    elif change == 'input_events':
        report['inputs_used']['E']['events'] += 1
    elif change == 'extra_input_change':
        report['inputs_used']['unexpected_change'] = True
    elif change == 'unbound_control':
        report['inputs_used']['input_order_control'] = {'different': True}
    elif change == 'event_totals':
        report['drive']['external_event_totals']['I'] += 1
    else:
        name = next(iter(original['code']))
        report['code'][name] = 'b'*64
    with pytest.raises(ValueError):
        temporal.validate_temporal_case(report, original, protocol, SHA, SHA, .025, 'cue')


def fields():
    counts = np.zeros((1000, 17), dtype=np.int64)
    counts[21, 5] = 3
    counts[888, 16] = 7
    return dict(field_counts=counts, rates_hz=counts/(temporal.FIELD_SIZES[None, :]*.005),
        field_unit_counts=temporal.FIELD_SIZES.copy(), field_indices=np.arange(17),
        time_start_ms=np.arange(1000)*5.)


@pytest.mark.parametrize('change', ['rate', 'negative', 'field_order', 'time_order', 'sizes'])
def test_field_units_counts_and_exact_axes_are_not_silently_coerced(change):
    arrays = fields()
    counts, rates = temporal.validate_fields(arrays)
    assert counts[21, 5] == 3 and rates[21, 5] == 3/25
    if change == 'rate':
        arrays['rates_hz'][21, 5] += .01
    elif change == 'negative':
        arrays['field_counts'][0, 0] = -1
    elif change == 'field_order':
        arrays['field_indices'] = arrays['field_indices'][::-1]
    elif change == 'time_order':
        arrays['time_start_ms'] = arrays['time_start_ms'][::-1]
    else:
        arrays['field_unit_counts'][0] += 1
    with pytest.raises(ValueError):
        temporal.validate_fields(arrays)


def histogram():
    expected = np.empty((3, 1, 5, 3, 17), dtype=np.int64)
    for pi, (start, stop) in enumerate(temporal.PHASES.values()):
        for wi, width in enumerate(temporal.WINDOWS_MS):
            expected[:, :, pi, wi] = round((stop-start)/width)*temporal.FIELD_SIZES
    indices = np.indices(expected.shape).reshape(5, -1).T
    return dict(coordinates=np.c_[indices, np.zeros(len(indices), dtype=np.int64)],
        observations=expected.reshape(-1).copy(), coordinate_axes=np.asarray(temporal.pfc_count_histograms.AXES),
        condition_names=np.asarray(temporal.CONDITIONS), source_rng_seeds=np.asarray([temporal.SEED]),
        phase_names=np.asarray(list(temporal.PHASES)), phase_bounds_ms=np.asarray(list(temporal.PHASES.values())),
        window_ms=np.asarray(temporal.WINDOWS_MS), field_unit_counts=temporal.FIELD_SIZES.copy(),
        expected_observations=expected)


@pytest.mark.parametrize('change', ['condition_order', 'phase_order', 'field_sizes', 'axes', 'mass', 'duplicate'])
def test_histogram_preserves_named_axes_and_complete_integer_denominators(change):
    arrays = histogram()
    temporal.validate_histogram(arrays)
    if change == 'condition_order':
        arrays['condition_names'] = arrays['condition_names'][::-1]
    elif change == 'phase_order':
        arrays['phase_names'] = arrays['phase_names'][::-1]
    elif change == 'field_sizes':
        arrays['field_unit_counts'][0] += 1
    elif change == 'axes':
        arrays['coordinate_axes'] = arrays['coordinate_axes'][::-1]
    elif change == 'mass':
        arrays['observations'][0] -= 1
    else:
        arrays['coordinates'] = np.r_[arrays['coordinates'], arrays['coordinates'][:1]]
        arrays['observations'] = np.r_[arrays['observations'], arrays['observations'][:1]]
    with pytest.raises(ValueError):
        temporal.validate_histogram(arrays)


def test_support_extension_keeps_zero_mass_and_signed_complete_cdf_difference():
    original = histogram()
    reversed_histogram = copy.deepcopy(original)
    reversed_histogram['observations'][0] -= 1
    reversed_histogram['coordinates'] = np.r_[
        reversed_histogram['coordinates'], np.asarray([[0, 0, 0, 0, 0, 16]])]
    reversed_histogram['observations'] = np.r_[reversed_histogram['observations'], 1]
    temporal.validate_histogram(reversed_histogram)
    old_support = np.arange(16)
    support = temporal.joint_support(old_support, [original, reversed_histogram])
    assert support.tolist() == list(range(17))
    old_mass = temporal.histogram_masses(original, support)
    new_mass = temporal.histogram_masses(reversed_histogram, support)
    difference = new_mass-old_mass
    assert difference.dtype == np.int64
    assert difference[0, 0, 0, 0, 0] == -1 and difference[0, 0, 0, 0, 16] == 1
    assert np.count_nonzero(difference) == 2
    assert np.all(difference[..., 1:16] == 0)
    shape = (3, 1, 5, 3, 17)
    original_cdf = temporal.pfc_count_histograms.cdf(original['coordinates'], original['observations'], old_support, shape)[:, 0]
    extended = temporal.comparison.cdf_on_support(original_cdf, old_support, support)
    changed = temporal.pfc_count_histograms.cdf(reversed_histogram['coordinates'], reversed_histogram['observations'], support, shape)[:, 0]
    assert np.all(extended[..., -1] == changed[..., -1])
    assert changed[0, 0, 0, 0, 0]-extended[0, 0, 0, 0, 0] == pytest.approx(-1/100000)
    assert np.all(changed[..., -1] == 1.)


def test_same_clock_pairs_retain_all_conditions_signed_sources_and_paired_controls():
    field_levels, response_levels, refs = {}, {}, {}
    differences = [0., 2., -3.]
    for clock_index, (label, dt) in enumerate(temporal.CLOCKS):
        original_values = np.empty((3, 5, 8))
        reversed_values = np.empty_like(original_values)
        field_levels['original_'+label], field_levels['reversed_'+label] = {}, {}
        for ci, condition in enumerate(temporal.CONDITIONS):
            old, delta = 100*(clock_index+1)+ci, differences[ci]*(clock_index+1)
            original_values[ci], reversed_values[ci] = old, old+delta
            field_levels['original_'+label][condition] = np.full((1000, 17), old, dtype=float)
            field_levels['reversed_'+label][condition] = np.full((1000, 17), old+delta, dtype=float)
            refs[(dt, condition)] = dict(case_id=f'pfc_{temporal.SEED}_{condition}',
                                        original_report_sha256=SHA, reversed_report_sha256='b'*64)
        response_levels['original_'+label], response_levels['reversed_'+label] = original_values, reversed_values
    field_rows = temporal.comparison.discrepancy_rows({temporal.SCHEME: field_levels}, temporal.PAIRS)
    source, contrasts = temporal.comparison.response_rows({temporal.SCHEME: response_levels}, temporal.PAIRS)
    assert len(field_rows) == 510 and len(source) == len(contrasts) == 240
    bound_source = temporal.bind_rows(source, refs, SHA)
    bound_contrasts = temporal.bind_rows(contrasts, refs, SHA)
    assert {row['condition'] for row in bound_source} == set(temporal.CONDITIONS)
    assert {row['signed_mean_hz'] for row in field_rows if row['condition'] == 'cue'} == {2., 4.}
    assert {row['left_minus_right'] for row in bound_source if row['condition'] == 'strong_cue'} == {-3., -6.}
    assert {row['left_minus_right'] for row in bound_contrasts if row['contrast'] == 'strong_cue_minus_cue'} == {-5., -10.}
    assert all(row['original_case_report_sha256'] == SHA+';'+SHA for row in bound_contrasts)
    wrong = dict(source[0], right_cohort='original_'+temporal.CLOCKS[1][0])
    with pytest.raises(ValueError, match='SAME_CLOCK_PAIRING'):
        temporal.bind_rows([wrong], refs, SHA)


def test_bound_primitive_cannot_be_replaced_by_an_unbound_file(tmp_path):
    directory = tmp_path/'analysis'
    directory.mkdir()
    path = directory/'phase_responses.npz'
    np.savez(path, synthetic_values=np.asarray([1., 2.]))
    record = dict(bytes=path.stat().st_size, sha256=temporal.sha256_file(path))
    report = dict(outputs={path.name: record})
    manifest = {'analysis/'+path.name: dict(path='analysis/'+path.name, **record)}
    assert temporal.bound_output(tmp_path, manifest, directory.relative_to(tmp_path), report, path.name) == path
    np.savez(path, synthetic_values=np.asarray([3., 4.]))
    with pytest.raises(ValueError, match='hash mismatch'):
        temporal.bound_output(tmp_path, manifest, directory.relative_to(tmp_path), report, path.name)


def test_complete_cdf_rows_use_same_clock_sign_and_keep_omission():
    support = np.asarray([0, 1])
    levels = {}
    for index, (label, _) in enumerate(temporal.CLOCKS):
        levels['original_'+label] = np.broadcast_to(
            [.25, 1.], (3, 5, 3, 17, 2)).copy()
        levels['reversed_'+label] = np.broadcast_to(
            [.25+(index+1)*.125, 1.], (3, 5, 3, 17, 2)).copy()
    rows = temporal.comparison.count_cdf_rows({temporal.SCHEME: levels}, support, temporal.PAIRS)
    assert len(rows) == 3060
    assert {row['condition'] for row in rows} == set(temporal.CONDITIONS)
    assert {row['left_minus_right'] for row in rows if row['count_threshold'] == 0} == {.125, .25}
    assert all(row['left_minus_right'] == 0 for row in rows if row['count_threshold'] == 1)
