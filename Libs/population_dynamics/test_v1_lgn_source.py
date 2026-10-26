"""Inventory-bound LGN extraction without unrelated source payloads."""
import json
import io
import zipfile

import pytest
import h5py
import numpy as np
import pandas as pd

from .provenance import sha256_file
from .v1_lgn_source import (SOURCE_MEMBERS, extract_tables, read_node_columns,
    read_source_cells, exact_column_comparison, DOMINANT_FIELDS, DUAL_FIELDS, DUAL_MODELS,
    CONSTRUCTION_FILES, read_input_construction_cells, extract_complete_members)


def fixture(tmp_path, mutation=None):
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as sink:
        for name in SOURCE_MEMBERS:
            sink.writestr(name, 'id model_id x y\n0 sON_TF8 1.0 2.0\n')
        sink.writestr('unrelated-data', 'Do not extract me.')
    with zipfile.ZipFile(archive) as source:
        rows = [dict(path=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                     crc32=f'{i.CRC:08x}') for i in source.infolist()]
    if mutation == 'crc': rows[0]['crc32'] = 'ffffffff'
    if mutation == 'missing': rows.pop(0)
    inventory = tmp_path/'inventory'; inventory.mkdir()
    path = inventory/'dependency_inventory.json'
    path.write_text(json.dumps(dict(archive_bytes=archive.stat().st_size, archive_members=rows)))
    (inventory/'delivery.json').write_text(json.dumps(dict(exit_code=0, files=[dict(
        path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))])))
    return archive, inventory


def test_only_selected_members_are_extracted_with_exact_bytes(tmp_path):
    archive, inventory = fixture(tmp_path)
    report = extract_tables(archive, inventory, tmp_path/'result')
    assert report['status'] == 'SOURCE_LGN_TABLES_EXTRACTED'
    assert {p.name for p in (tmp_path/'result').iterdir()} == {'intake.json', *map(lambda n:n.rsplit('/',1)[-1], SOURCE_MEMBERS)}
    assert len(report['extracted']) == 2
    for row in report['extracted']:
        assert row['sha256'] == sha256_file(tmp_path/'result'/row['file'])


@pytest.mark.parametrize('mutation', ['crc', 'missing'])
def test_changed_or_missing_source_member_is_rejected(tmp_path, mutation):
    archive, inventory = fixture(tmp_path, mutation)
    with pytest.raises(ValueError, match='MEMBER'):
        extract_tables(archive, inventory, tmp_path/'result')
    assert json.loads((tmp_path/'result/intake.json').read_text())['status'] == 'FAILED_SOURCE_LGN_TABLE_EXTRACTION'


def test_changed_inventory_receipt_is_rejected_before_extraction(tmp_path):
    archive, inventory = fixture(tmp_path)
    with (inventory/'dependency_inventory.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError, match='RECEIPT_CHANGED'):
        extract_tables(archive, inventory, tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_node_group_index_is_joined_before_sorting_source_ids(tmp_path):
    nodes, types = tmp_path/'nodes.h5', tmp_path/'types.csv'
    with h5py.File(nodes, 'w') as file:
        group = file.create_group('nodes/lgn')
        group['node_id'] = [1, 0]
        group['node_type_id'] = [8, 7]
        group['node_group_id'] = [0, 0]
        group['node_group_index'] = [1, 0]
        group['0/x'] = [10., 20.]
        group['0/y'] = [11., 21.]
    types.write_text('node_type_id pop_name\n7 sON_TF8\n8 sOFF_TF8\n')
    frame, _ = read_node_columns(nodes, types, 'lgn')
    assert frame.index.tolist() == [0, 1]
    assert frame['x'].tolist() == [10., 20.]
    assert frame['model_id'].tolist() == ['sON_TF8', 'sOFF_TF8']


def test_type_code_relabeling_does_not_imply_model_identity_change():
    old = pd.DataFrame(dict(node_type_id=[0, 1], model_id=['sON_TF8', 'sON_TF4']), index=[0, 1])
    current = old.copy();current['node_type_id'] = [0, 100]
    result = exact_column_comparison(old, current, ['node_type_id', 'model_id'])
    assert not result['node_type_id']['exact'] and result['node_type_id']['mismatches'] == 1
    assert result['model_id']['exact']


def test_exact_coordinate_comparison_does_not_fit_tolerance():
    first = pd.DataFrame(dict(x=[1.0]), index=[0]);second = first.copy()
    second.loc[0, 'x'] = np.nextafter(1., 2.)
    assert exact_column_comparison(first, second, ['x'])['x']['mismatches'] == 1


def test_source_inactive_nan_fields_remain_nan_but_active_nan_is_rejected(tmp_path):
    data = dict(id=[0,1], model_id=['sON_TF8','sONsOFF_001'])
    for name in DOMINANT_FIELDS:data[name] = [1., 1.]
    for name in DUAL_FIELDS:data[name] = [np.nan, 1.]
    cells, models = tmp_path/'cells.csv', tmp_path/'models.csv'
    frame = pd.DataFrame(data);frame.to_csv(cells, sep=' ', index=False, na_rep='nan')
    models.write_text('model_id level_of_detail\nsON_TF8 filter\nsONsOFF_001 filter\n')
    loaded, _ = read_source_cells(cells, models)
    assert loaded.loc[0, list(DUAL_FIELDS)].isna().all()
    frame.loc[1, DUAL_FIELDS[0]] = np.nan;frame.to_csv(cells, sep=' ', index=False, na_rep='nan')
    with pytest.raises(ValueError, match='FILTER_VALUE_INVALID'):
        read_source_cells(cells, models)


def test_grouped_source_properties_preserve_mapping_and_structural_absence(tmp_path):
    nodes, types = tmp_path/'groups.h5', tmp_path/'types.csv'
    with h5py.File(nodes, 'w') as file:
        group = file.create_group('nodes/lgn')
        group['node_id'] = [2, 0, 1]
        group['node_type_id'] = [8, 7, 7]
        group['node_group_id'] = [1, 0, 0]
        group['node_group_index'] = [0, 1, 0]
        group['0/x'] = [10., 20.];group['0/y'] = [11., 21.]
        group['1/x'] = [30.];group['1/y'] = [31.]
        group['1/tuning_angle'] = [45.]
    types.write_text('node_type_id pop_name\n7 sON_TF8\n8 sONsOFF_001\n')
    frame, inventory = read_node_columns(nodes, types, 'lgn')
    assert frame['x'].tolist() == [20., 10., 30.]
    assert inventory['property_groups']['tuning_angle'] == [1]
    source = frame.copy();source.loc[[0,1], 'tuning_angle'] = [99., 98.]
    availability = {'tuning_angle': frame['node_group_id'].isin([1]).to_numpy()}
    result = exact_column_comparison(source, frame, ['tuning_angle'], availability)['tuning_angle']
    assert result['exact'] and result['values'] == 1 and result['unavailable_rows'] == 2
    assert result['source_nan_at_unavailable_rows'] == 0
    with h5py.File(nodes, 'r+') as file:file['nodes/lgn/node_group_index'][0] = 1
    with pytest.raises(ValueError, match='PROPERTY_INDEX_INVALID'):
        read_node_columns(nodes, types, 'lgn')


@pytest.mark.parametrize('damage', [None, 'crc', 'required_member_truncation'])
def test_stream_member_crc_and_completion_boundaries(tmp_path, damage):
    class Stream(io.BytesIO):
        def seek(self, *args):
            raise OSError('Non-seekable source stream')
    stream = Stream()
    payload = b'SOURCE_MEMBER_CONTENT_FOR_NATIVE_CRC_TEST\n'
    tail = b'Z' * 4096
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('needed.py', payload)
        archive.writestr('unused.bin', tail)
    raw = stream.getvalue()
    offset = raw.index(payload)
    prefix = raw[:raw.index(tail) + 16]
    if damage == 'crc':
        prefix = prefix[:offset] + b'X' + prefix[offset + 1:]
    elif damage == 'required_member_truncation':
        prefix = prefix[:offset + len(payload)//2]
    source = tmp_path/'source.zip.part';source.write_bytes(prefix)
    if damage is None:
        extract_complete_members(source, tmp_path/'selected', ('needed.py',))
        assert (tmp_path/'selected/needed.py').read_bytes() == payload
        assert not (tmp_path/'selected/unused.bin').exists()
    else:
        with pytest.raises(ValueError, match='SOURCE_RUNTIME_MEMBER_READ_FAILED'):
            extract_complete_members(source, tmp_path/'selected', ('needed.py',))


CONSTRUCTED_MODEL_NAMES = ('sON_TF1', 'sON_TF2', 'sON_TF4', 'sON_TF8', 'sOFF_TF1',
    'sOFF_TF2', 'sOFF_TF4', 'sOFF_TF8', 'sOFF_TF15', 'tOFF_TF4', 'tOFF_TF8',
    'tOFF_TF15', 'sONsOFF_001', 'sONtOFF_001')


def _bind_construction_files(archive, mode="canary"):
    rows = [dict(path=name, bytes=(archive/name).stat().st_size, sha256=sha256_file(archive/name))
            for name in CONSTRUCTION_FILES]
    status = {'canary': 'PASS_SOURCE_INPUT_CANARY', 'full': 'PASS_SOURCE_INPUT_BUILD_REPEATABILITY'}[mode]
    (archive/'delivery.json').write_text(json.dumps(dict(status=status,
        exit_code=0, mode=mode, files=rows)))


def _construction_fixture(tmp_path, mode="canary"):
    archive = tmp_path/'construction'; directory = archive/'primary/glif'; directory.mkdir(parents=True)
    ids = np.arange(17400, dtype=np.uint64)
    names = np.asarray(CONSTRUCTED_MODEL_NAMES)[ids.astype(np.int64) % 14]
    with h5py.File(directory/'lgn_nodes.h5', 'w') as handle:
        group = handle.create_group('nodes/lgn')
        group['node_id'] = ids[::-1]
        group['node_type_id'] = (100 + ids % 14)[::-1]
        group['node_group_id'] = (ids % 2)[::-1]
        group['node_group_index'] = (ids // 2)[::-1]
        for identity in (0, 1):
            selected = ids % 2 == identity
            values = ids[selected].astype(np.float64) + .125
            dual = np.isin(names[selected], list(DUAL_MODELS))
            for field in DOMINANT_FIELDS:
                group[f'{identity}/{field}'] = values
            for field in DUAL_FIELDS:
                group[f'{identity}/{field}'] = np.where(dual, values + .5, np.nan)
    (directory/'lgn_node_types.csv').write_text('node_type_id pop_name level_of_detail model_type\n' +
        ''.join(f'{100+i} {name} filter virtual\n' for i, name in enumerate(CONSTRUCTED_MODEL_NAMES)))
    with h5py.File(directory/'bkg_nodes.h5', 'w') as handle:
        group = handle.create_group('nodes/bkg')
        for key, value in [('node_id', 0), ('node_type_id', 100), ('node_group_id', 0), ('node_group_index', 0)]:
            group[key] = [value]
        group['0/x'] = [0.]; group['0/y'] = [0.]
    (directory/'bkg_node_types.csv').write_text('node_type_id pop_name model_type\n100 SG_001 virtual\n')
    model_counts = {name: int(np.count_nonzero(names == name)) for name in CONSTRUCTED_MODEL_NAMES}
    report = dict(status='ORIGINAL_CIRCUIT_INPUTS_COMPLETE', exit_code=0, mode=mode, validation=dict(
        lgn=dict(source_units=17400, model_counts=model_counts,
                 intentional_nan_values=int(np.count_nonzero(~np.isin(names, list(DUAL_MODELS))))*len(DUAL_FIELDS)),
        bkg=dict(source_units=1)))
    (archive/'primary/report.json').write_text(json.dumps(report)); _bind_construction_files(archive, mode)
    return archive


@pytest.mark.parametrize("mode", ["canary", "full"])
def test_constructed_cells_use_group_indices_sort_ids_and_preserve_explicit_nans(tmp_path, mode):
    archive = _construction_fixture(tmp_path, mode)
    cells, models, binding = read_input_construction_cells(archive)
    assert cells.index.tolist() == list(range(17400))
    assert np.array_equal(cells['x'].to_numpy(), np.arange(17400)+.125)
    assert cells.loc[13, 'model_id'] == 'sONtOFF_001'
    assert cells.loc[13, 'kpeaks_non_dom_0'] == 13.625
    assert cells.loc[0, list(DUAL_FIELDS)].isna().all()
    assert len(models) == 14 and set(models['model_id']) == set(CONSTRUCTED_MODEL_NAMES)
    assert binding['mode'] == mode
    assert set(binding) == {'delivery_sha256', 'mode', 'files'}
    assert set(binding['files']) == set(CONSTRUCTION_FILES)
    assert str(archive) not in json.dumps(binding)


@pytest.mark.parametrize('field, row, value, message', [
    ('x', 0, np.nan, 'FILTER_VALUE_INVALID'),
    ('kpeaks_non_dom_0', 6, np.nan, 'FILTER_VALUE_INVALID'),
    ('kpeaks_non_dom_0', 0, 1., 'MONO_NAN_SEMANTICS'),
])
def test_constructed_filter_semantics_are_validated_after_receipt_binding(tmp_path, field, row, value, message):
    archive = _construction_fixture(tmp_path)
    with h5py.File(archive/'primary/glif/lgn_nodes.h5', 'r+') as handle:
        handle[f'nodes/lgn/0/{field}'][row] = value
    _bind_construction_files(archive)
    with pytest.raises(ValueError, match=message):
        read_input_construction_cells(archive)


def test_constructed_background_files_are_bound_even_for_lgn_rate_input(tmp_path):
    archive = _construction_fixture(tmp_path)
    with (archive/'primary/glif/bkg_node_types.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='ARCHIVE_CHANGED'):
        read_input_construction_cells(archive)


def test_failed_input_construction_cannot_supply_lgn_cells(tmp_path):
    archive = _construction_fixture(tmp_path)
    path = archive/'primary/report.json'; report = json.loads(path.read_text()); report['exit_code'] = 1
    path.write_text(json.dumps(report)); _bind_construction_files(archive)
    with pytest.raises(ValueError, match='COMPLETED_INPUT_CONSTRUCTION'):
        read_input_construction_cells(archive)


@pytest.mark.parametrize('mode,status', [
    ('full', 'PASS_SOURCE_INPUT_FULL'),
    ('full', 'PASS_SOURCE_INPUT_CANARY'),
    ('canary', 'PASS_SOURCE_INPUT_BUILD_REPEATABILITY'),
    ('full', 'FAILED_SOURCE_INPUT_BUILD'),
])
def test_input_construction_status_must_match_actual_producer_mode(tmp_path, mode, status):
    archive = _construction_fixture(tmp_path, mode)
    path = archive / 'delivery.json'
    delivery = json.loads(path.read_text())
    delivery['status'] = status
    path.write_text(json.dumps(delivery))
    with pytest.raises(ValueError, match='COMPLETED_INPUT_CONSTRUCTION'):
        read_input_construction_cells(archive)


@pytest.mark.parametrize('field,value', [('exit_code', 1), ('mode', 'canary')])
def test_full_input_delivery_must_be_successful_and_match_primary_mode(tmp_path, field, value):
    archive = _construction_fixture(tmp_path, 'full')
    path = archive / 'delivery.json'
    delivery = json.loads(path.read_text())
    delivery[field] = value
    path.write_text(json.dumps(delivery))
    with pytest.raises(ValueError, match='COMPLETED_INPUT_CONSTRUCTION'):
        read_input_construction_cells(archive)


def test_full_input_report_rng_metadata_remains_content_bound(tmp_path):
    archive = _construction_fixture(tmp_path, 'full')
    path = archive / 'primary/report.json'
    report = json.loads(path.read_text())
    report['rng'] = {'lgn_selection_python_seed': 999}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='REPLAY_RECEIPT_CHANGED'):
        read_input_construction_cells(archive)
