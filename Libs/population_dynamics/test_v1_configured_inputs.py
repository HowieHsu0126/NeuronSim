"""Configured-source identity, multiplicity and parser regressions."""
import h5py
import numpy as np
import pytest
from .v1_configured_inputs import compare_population, validate_plan, read_csv_events, read_input_inventory, read_h5_events


def plan():
    return dict(population=np.array([0, 0, 0, 1]), source_id=np.array([2, 0, 1, 0]),
                source_counts=np.array([3, 0, 1, 1]),
                source_event_unit=np.array([0, 0, 0, 2, 3]),
                source_event_time_ms=np.array([.052912, 2., 2., 4., 3.]))


def test_population_join_preserves_unsorted_ids_duplicates_and_silent_units():
    result = compare_population(plan(), 0, np.array([1, 2, 2, 2]), np.array([4., 2., .052912, 2.]))
    assert result['all_event_pairs_exact']
    assert result['mapped_generators'] == 3 and result['silent_source_generators'] == 1
    assert result['csv_active_source_ids'] == 2
    # Source ID zero belongs to both populations; a global join is invalid.
    assert compare_population(plan(), 1, np.array([0]), np.array([3.]))['all_event_pairs_exact']
    assert not compare_population(plan(), 0, np.array([0]), np.array([3.]))['all_event_pairs_exact']


def test_omitting_one_duplicate_is_a_count_failure():
    result = compare_population(plan(), 0, np.array([1, 2, 2]), np.array([4., 2., .052912]))
    assert not result['all_event_pairs_exact']
    assert result['mismatched_generator_counts'] == 1


def test_unknown_source_events_are_disclosed_and_preclude_equality():
    result = compare_population(plan(), 1, np.array([0, 99]), np.array([3., 3.]))
    assert not result['all_event_pairs_exact']
    assert result['unknown_csv_events'] == 1
    assert result['unknown_csv_source_ids'] == [dict(source_id=99, events=1)]


def test_one_ulp_time_difference_is_not_relaxed():
    result = compare_population(plan(), 1, np.array([0]), np.array([np.nextafter(3., 4.)]))
    assert not result['all_event_pairs_exact']
    assert result['timestamp_mismatches'] == 1


@pytest.mark.parametrize('fault', ['duplicate_identity', 'wrong_counts', 'unknown_event_unit'])
def test_corrupt_archive_plan_is_rejected(fault):
    value = plan()
    if fault == 'duplicate_identity': value['source_id'][0] = 0
    elif fault == 'wrong_counts': value['source_counts'][0] = 2
    else: value['source_event_unit'][0] = 100
    with pytest.raises(ValueError, match='PLAN'):
        validate_plan(value)


def test_csv_keeps_file_scoped_population_label_and_duplicate_events(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_text('timestamps population node_ids\n1.25 lgn 0\n1.25 lgn 0\n2 lgn 0\n')
    ids, times, labels = read_csv_events(path, 3.)
    assert ids.tolist() == [0, 0, 0] and times.tolist() == [1.25, 1.25, 2.]
    assert labels.tolist() == ['lgn']*3


@pytest.mark.parametrize('text', [
    'time population node_ids\n1 lgn 0\n',
    'timestamps population node_ids\n1 lgn 0.5\n',
    'timestamps population node_ids\nnan lgn 0\n',
    'timestamps population node_ids\n0 lgn 0\n',
    'timestamps population node_ids\n4 lgn 0\n',
    'timestamps population node_ids\n1 lgn 0\n2 bkg 0\n'])
def test_csv_invalid_fields_or_combined_populations_fail(tmp_path, text):
    path = tmp_path / 'source.csv'; path.write_text(text)
    with pytest.raises(ValueError, match='CSV'):
        read_csv_events(path, 3.)


def test_node_inventory_retains_silent_nodes_and_source_type_ids(tmp_path):
    nodes, types = tmp_path/'nodes.h5', tmp_path/'types.csv'
    with h5py.File(nodes, 'w') as source:
        group = source.create_group('nodes/lgn')
        group['node_id'] = [1, 0, 2]
        group['node_type_id'] = [11, 10, 10]
    types.write_text('node_type_id model_type\n10 virtual\n11 virtual\n')
    ids, report = read_input_inventory(nodes, types, 'lgn')
    assert ids.tolist() == [1, 0, 2] and report['source_nodes'] == 3
    assert report['observed_node_types'] == 2
    with pytest.raises(ValueError, match='NODE_SET'):
        read_input_inventory(nodes, types, 'bkg')


@pytest.mark.parametrize('grouped', [False, True])
def test_h5_preserves_layout_population_and_repeated_times(tmp_path, grouped):
    path = tmp_path/'source.h5'
    with h5py.File(path, 'w') as source:
        group = source.create_group('spikes/lgn' if grouped else 'spikes')
        group['node_ids' if grouped else 'gids'] = [2, 0, 2]
        group['timestamps'] = [1.25, 2., 1.25]
        if grouped: group['timestamps'].attrs['units'] = 'ms'
    ids, times, metadata = read_h5_events(path, 3.)
    assert ids.tolist() == [2, 0, 2] and times.tolist() == [1.25, 2., 1.25]
    assert metadata['source_population_labels'] == (['lgn'] if grouped else [])
    assert metadata['declared_time_units'] == ('ms' if grouped else None)


@pytest.mark.parametrize('fault', ['units', 'population', 'identity', 'times'])
def test_h5_rejects_ambiguous_population_units_and_invalid_values(tmp_path, fault):
    path = tmp_path/'source.h5'
    with h5py.File(path, 'w') as source:
        group = source.create_group('spikes/lgn')
        group['node_ids'] = [0.5] if fault == 'identity' else [0]
        group['timestamps'] = [np.nan] if fault == 'times' else [1.25]
        if fault == 'units': group['timestamps'].attrs['units'] = 's'
        if fault == 'population': source.create_group('spikes/bkg')
    with pytest.raises(ValueError, match='H5'):
        read_h5_events(path, 3.)
