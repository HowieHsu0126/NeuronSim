"""Extract the two inventoried source LGN tables for visual-input reconstruction."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pandas as pd

from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import bound_json
from .v1_configured_inputs import delivered_file, read_input_inventory

SOURCE_MEMBERS = ('GLIF_network2/lgn_full_col_cells_3.csv',
                  'GLIF_network2/lgn_full_col_cell_models_3.csv')


def extract_tables(archive_path: Path, inventory_archive: Path, output: Path) -> dict:
    inventory, delivery = bound_json(inventory_archive, 'dependency_inventory.json')
    if delivery['exit_code'] != 0 or archive_path.stat().st_size != inventory['archive_bytes']:
        raise ValueError('V1_LGN_COMPLETE_SOURCE_ARCHIVE_REQUIRED')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='FAILED_SOURCE_LGN_TABLE_EXTRACTION', source_archive=str(archive_path),
                  inventory_sha256=sha256_file(inventory_archive/'dependency_inventory.json'),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source_simulations=0, production_episodes=0, source_fidelity_qualified=False,
                  extracted=[])
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            for name in SOURCE_MEMBERS:
                rows = [row for row in inventory['archive_members'] if row['path'] == name]
                if len(rows) != 1 or names.count(name) != 1:
                    raise ValueError('V1_LGN_SOURCE_MEMBER_NOT_UNIQUE:' + name)
                row = rows[0]; info = archive.getinfo(name)
                if (info.file_size != row['bytes'] or info.compress_size != row['compressed_bytes']
                        or f'{info.CRC:08x}' != row['crc32']):
                    raise ValueError('V1_LGN_SOURCE_MEMBER_DIFFERS_FROM_INVENTORY:' + name)
                target = output/Path(name).name
                with archive.open(name) as source, target.open('xb') as sink:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        sink.write(block)
                if target.stat().st_size != row['bytes']:
                    raise ValueError('V1_LGN_SOURCE_MEMBER_SIZE_DIFFERS:' + name)
                with target.open(encoding='utf-8') as stream:
                    header = stream.readline().rstrip('\n')
                    first_row = stream.readline().rstrip('\n')
                result['extracted'].append(dict(member=name, file=target.name,
                    bytes=target.stat().st_size, sha256=sha256_file(target),
                    crc32=row['crc32'], header=header, first_row=first_row))
        result['status'] = 'SOURCE_LGN_TABLES_EXTRACTED'
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        freeze_json(output/'intake.json', result)
    return result



DOMINANT_FIELDS = ('x', 'y', 'spatial_size', 'kpeaks_dom_0', 'kpeaks_dom_1',
                   'weight_dom_0', 'weight_dom_1', 'delay_dom_0', 'delay_dom_1')
DUAL_FIELDS = ('kpeaks_non_dom_0', 'kpeaks_non_dom_1', 'weight_non_dom_0',
              'weight_non_dom_1', 'delay_non_dom_0', 'delay_non_dom_1', 'tuning_angle', 'sf_sep')
DUAL_MODELS = {'sONsOFF_001', 'sONtOFF_001'}


def read_source_cells(cells_path, models_path):
    cells = pd.read_csv(cells_path, sep=' ', float_precision='round_trip')
    models = pd.read_csv(models_path, sep=' ')
    return _validate_source_cells(cells, models), models


def _validate_source_cells(cells, models):
    """Validate source filter values without filling inactive fields."""
    needed = {'id', 'model_id', *DOMINANT_FIELDS, *DUAL_FIELDS}
    if not needed.issubset(cells) or not {'model_id', 'level_of_detail'}.issubset(models):
        raise ValueError('V1_LGN_SOURCE_TABLE_FIELDS_MISSING')
    ids = cells['id'].to_numpy()
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError('V1_LGN_SOURCE_TABLE_IDS_INVALID')
    if (models['model_id'].duplicated().any() or not (models['level_of_detail'] == 'filter').all()
            or set(cells['model_id']) != set(models['model_id'])):
        raise ValueError('V1_LGN_SOURCE_MODEL_IDENTITIES_DIFFER')
    dual = cells['model_id'].isin(DUAL_MODELS)
    for fields, mask in [(DOMINANT_FIELDS, np.ones(len(cells), dtype=bool)), (DUAL_FIELDS, dual)]:
        for name in fields:
            values = cells[name].to_numpy()
            if values.dtype.kind not in 'fiu' or not np.isfinite(values[mask]).all():
                raise ValueError('V1_LGN_REQUIRED_SOURCE_FILTER_VALUE_INVALID:' + name)
    return cells.set_index('id').sort_index()


def read_node_columns(nodes_path, types_path, node_set):
    ids, inventory = read_input_inventory(nodes_path, types_path, node_set)
    types = {int(row['node_type_id']): row for row in inventory['type_records']}
    frames, property_groups, omitted = [], {}, {}
    with h5py.File(nodes_path, 'r') as source:
        group = source['nodes'][node_set]
        group_ids, indices = group['node_group_id'][:], group['node_group_index'][:]
        if (group_ids.shape != ids.shape or indices.shape != ids.shape
                or group_ids.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
                or np.any(group_ids < 0) or np.any(indices < 0)):
            raise ValueError('V1_LGN_SOURCE_METADATA_GROUP_MAPPING_INVALID')
        type_ids = group['node_type_id'][:]
        for group_id in np.unique(group_ids):
            selected = np.flatnonzero(group_ids == group_id)
            label = str(int(group_id))
            if label not in group:
                raise ValueError('V1_LGN_SOURCE_METADATA_GROUP_MISSING:' + label)
            properties = group[label]
            if not {'x', 'y'}.issubset(properties):
                raise ValueError('V1_LGN_SOURCE_POSITION_FIELDS_MISSING:' + label)
            row_indices = indices[selected]
            columns = dict(id=ids[selected], node_type_id=type_ids[selected],
                           node_group_id=group_ids[selected],
                           model_id=[types[int(i)]['pop_name'] for i in type_ids[selected]])
            for name, values in properties.items():
                if not isinstance(values, h5py.Dataset) or values.ndim != 1:
                    omitted.setdefault(label, []).append(name)
                    continue
                if np.any(row_indices >= len(values)):
                    raise ValueError('V1_LGN_SOURCE_PROPERTY_INDEX_INVALID:' + label + '/' + name)
                columns[name] = values[:][row_indices.astype(np.int64)]
                property_groups.setdefault(name, []).append(int(group_id))
            frames.append(pd.DataFrame(columns).set_index('id'))
    # Outer alignment represents structural absence, not a supplied NaN value.
    # Presence is retained below and those rows are excluded from value equality.
    inventory['property_groups'] = property_groups
    inventory['non_scalar_per_node_properties_not_compared'] = omitted
    return pd.concat(frames, axis=0, sort=False).sort_index(), inventory


CONSTRUCTION_FILES = ('primary/report.json', 'primary/glif/lgn_nodes.h5',
                      'primary/glif/lgn_node_types.csv', 'primary/glif/bkg_nodes.h5',
                      'primary/glif/bkg_node_types.csv')


def read_input_construction_cells(archive: Path):
    """Read the complete constructed LGN directly, with its background binding.

    The returned binding contains content identities only. A new realization
    with numerically identical node IDs is not interchangeable with this one.
    No intermediate CSV, rounding, imputation, or source model renaming is used.
    """
    archive = Path(archive)
    report, delivery = bound_json(archive, 'primary/report.json')
    mode = report.get('mode')
    # The full-build producer completes only after its independent replay agrees.
    completed_status = {'canary': 'PASS_SOURCE_INPUT_CANARY',
                        'full': 'PASS_SOURCE_INPUT_BUILD_REPEATABILITY'}
    if (delivery.get('exit_code') != 0 or report.get('exit_code') != 0
            or mode not in completed_status or delivery.get('mode') != mode
            or delivery.get('status') != completed_status[mode]
            or report.get('status') != 'ORIGINAL_CIRCUIT_INPUTS_COMPLETE'):
        raise ValueError('V1_LGN_COMPLETED_INPUT_CONSTRUCTION_REQUIRED')
    paths = {name: delivered_file(archive, name) for name in CONSTRUCTION_FILES[1:]}
    files = {name: {key: next(row for row in delivery['files'] if row['path'] == name)[key]
                    for key in ('bytes', 'sha256')} for name in CONSTRUCTION_FILES}
    cells, inventory = read_node_columns(paths['primary/glif/lgn_nodes.h5'],
                                        paths['primary/glif/lgn_node_types.csv'], 'lgn')
    background, background_inventory = read_node_columns(paths['primary/glif/bkg_nodes.h5'],
                                                         paths['primary/glif/bkg_node_types.csv'], 'bkg')
    models = pd.DataFrame(inventory['type_records'])
    models['model_id'] = models['pop_name']
    validation = report['validation']
    if (len(cells) != 17400 or len(models) != 14 or inventory['observed_node_types'] != 14
            or validation['lgn']['source_units'] != len(cells)
            or not (models['model_type'] == 'virtual').all()
            or len(background) != 1 or validation['bkg']['source_units'] != len(background)
            or background_inventory['declared_node_types'] != 1
            or any(row['model_type'] != 'virtual' for row in background_inventory['type_records'])):
        raise ValueError('V1_LGN_CONSTRUCTED_POPULATION_INVENTORY_DIFFERS')
    cells = _validate_source_cells(cells.reset_index(), models)
    for field in (*DOMINANT_FIELDS, *DUAL_FIELDS):
        groups = inventory['property_groups'].get(field, [])
        if not cells['node_group_id'].isin(groups).all():
            raise ValueError('V1_LGN_CONSTRUCTED_FILTER_FIELD_ABSENT:' + field)
    mono = ~cells['model_id'].isin(DUAL_MODELS)
    if not cells.loc[mono, list(DUAL_FIELDS)].isna().all().all():
        raise ValueError('V1_LGN_CONSTRUCTED_MONO_NAN_SEMANTICS_DIFFER')
    counts = {name: int(count) for name, count in cells['model_id'].value_counts().items()}
    if (counts != validation['lgn']['model_counts']
            or int(mono.sum()) * len(DUAL_FIELDS) != validation['lgn']['intentional_nan_values']):
        raise ValueError('V1_LGN_CONSTRUCTED_MODEL_COUNTS_DIFFER')
    for field in ('x', 'y'):
        values = background[field].to_numpy()
        if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('V1_LGN_CONSTRUCTED_BACKGROUND_POSITION_INVALID')
    binding = dict(delivery_sha256=sha256_file(archive/'delivery.json'), mode=mode, files=files)
    return cells, models, binding


def exact_column_comparison(first, second, columns, availability=None):
    if not np.array_equal(first.index.to_numpy(), second.index.to_numpy()):
        raise ValueError('V1_LGN_SOURCE_TABLE_IDS_DIFFER')
    result = {}
    for name in columns:
        full_a, full_b = first[name].to_numpy(), second[name].to_numpy()
        present = np.ones(len(full_a), dtype=bool) if availability is None or name not in availability else np.asarray(availability[name])
        if present.dtype != np.dtype(bool) or present.shape != full_a.shape:
            raise ValueError('V1_LGN_PROPERTY_AVAILABILITY_INVALID:' + name)
        a, b = full_a[present], full_b[present]
        same = a == b
        numeric = a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu'
        if numeric:
            same |= np.isnan(a) & np.isnan(b)
        row = dict(values=len(a), unavailable_rows=int((~present).sum()), mismatches=int((~same).sum()), exact=bool(same.all()))
        if numeric:
            row['source_nan_at_unavailable_rows'] = int(np.isnan(full_a[~present]).sum())
        if numeric:
            finite = np.isfinite(a) & np.isfinite(b)
            row['max_finite_absolute_difference'] = (float(np.max(np.abs(
                a[finite].astype(np.float64)-b[finite].astype(np.float64)))) if finite.any() else None)
        result[name] = row
    return result


def inspect_metadata(tables_archive, legacy_archive, current_archive, output):
    report = dict(status='FAILED_LGN_SOURCE_METADATA_INSPECTION', source_simulations=0,
                  production_episodes=0, source_fidelity_qualified=False,
                  original_visual_recipe_qualified=False, started_utc=datetime.now(timezone.utc).isoformat())
    try:
        intake, _ = bound_json(tables_archive, 'tables/intake.json')
        if intake['status'] != 'SOURCE_LGN_TABLES_EXTRACTED':
            raise ValueError('V1_LGN_EXTRACTED_SOURCE_TABLES_REQUIRED')
        cells_path = delivered_file(tables_archive, 'tables/lgn_full_col_cells_3.csv')
        models_path = delivered_file(tables_archive, 'tables/lgn_full_col_cell_models_3.csv')
        cells, models = read_source_cells(cells_path, models_path)
        builder_path = delivered_file(legacy_archive, 'generate_lgn_full_col.py/source_asset.txt')
        functions_path = delivered_file(legacy_archive, 'lgn_functions.py/source_asset.txt')
        legacy, legacy_inventory = read_node_columns(
            delivered_file(legacy_archive, 'lgn_nodes.h5/asset/lgn_nodes.h5'),
            delivered_file(legacy_archive, 'lgn_node_types.csv/asset/lgn_node_types.csv'), 'lgn')
        current, current_inventory = read_node_columns(
            delivered_file(current_archive, 'lgn_nodes.h5/asset/lgn_nodes.h5'),
            delivered_file(current_archive, 'lgn_node_types.csv/asset/lgn_node_types.csv'), 'lgn')
        if len(cells) != 17400 or len(models) != 14:
            raise ValueError('V1_LGN_FULL_SOURCE_DENOMINATOR_DIFFERS')
        legacy_columns = exact_column_comparison(cells, legacy, ['model_id', 'x', 'y'])
        common = sorted(set(cells.columns) & set(current.columns))
        availability = {name: current['node_group_id'].isin(groups).to_numpy()
                        for name, groups in current_inventory['property_groups'].items()}
        current_columns = exact_column_comparison(cells, current, common, availability)
        dual = cells['model_id'].isin(DUAL_MODELS).to_numpy()
        missing_required = {}
        for fields, required in [(DOMINANT_FIELDS, np.ones(len(cells), dtype=bool)), (DUAL_FIELDS, dual)]:
            for field in fields:
                available = availability.get(field, np.zeros(len(cells), dtype=bool))
                missing_required[field] = int((required & ~available).sum())
        raw_type_codes = exact_column_comparison(legacy, current, ['node_type_id'])
        report.update(source_builder_sha256=sha256_file(builder_path), source_functions_sha256=sha256_file(functions_path),
                      source_csv_coordinate_units='linear_degrees',
                      coordinate_unit_basis='Published builder labels X_len/Y_len in linear degrees; generate_positions_grids preserves that scale and the rate function uses node positions directly.',
                      source_cells_sha256=sha256_file(cells_path), source_models_sha256=sha256_file(models_path),
                      source_units=len(cells), source_models=len(models),
                      parser='Round-trip float64 CSV parsing; no rounding, filling or parameter fitting',
                      dual_subfield_units=int(cells['model_id'].isin(DUAL_MODELS).sum()),
                      inactive_dual_field_missing_values=int(cells.loc[~cells['model_id'].isin(DUAL_MODELS), list(DUAL_FIELDS)].isna().sum().sum()),
                      legacy_inventory=legacy_inventory, current_inventory=current_inventory,
                      source_to_legacy=legacy_columns, source_to_current_defined_fields=current_columns,
                      current_missing_required_values=missing_required,
                      current_required_fields_available=not any(missing_required.values()),
                      current_defined_values_exact=all(row['exact'] for row in current_columns.values()),
                      legacy_to_current_raw_type_codes=raw_type_codes,
                      required_filter_values_finite=True,
                      source_ids_types_positions_match_legacy=all(row['exact'] for row in legacy_columns.values()))
        report['status'] = ('SOURCE_LGN_TABLE_AND_LEGACY_METADATA_EXACT'
                            if report['source_ids_types_positions_match_legacy'] else 'SOURCE_LGN_METADATA_DIFFER')
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        freeze_json(output, report)
    return report



# Exact members needed by the published LGN functions, plus their source data/tests.
RUNTIME_MEMBERS = (
    '__init__.py', 'cellmodel.py', 'cursor.py', 'fitfuns.py', 'kernel.py',
    'lgnmodel1.py', 'linearfilter.py', 'lnunit.py', 'movie.py', 'nwb.py',
    'poissongeneration.py', 'spatialfilter.py', 'temporalfilter.py',
    'transferfunction.py', 'util_fns.py', 'utilities.py',
    'cell_data/tON_cell_data.csv', 'cell_data/sON_cell_data.csv',
    'cell_data/tOFF_cell_data.csv', 'cell_data/sOFF_cell_data.csv',
    'cell_data/sus_sus_cells_v3.csv', 'test/test_movie.py',
    'test/test_cellmodel.py', 'test/test_kernel.py',
    'test/test_linearfilter.py', 'test/test_poissongeneration.py',
)


def extract_complete_members(archive: Path, output: Path, members: tuple[str, ...]):
    """Read named members with libarchive CRC checks, stopping before unused data.

    A zero tool exit and all named regular files are required. This never repairs
    an archive, combines responses, or declares an incomplete archive complete.
    """
    tool = shutil.which('bsdtar')
    if tool is None:
        raise RuntimeError('SOURCE_RUNTIME_BSDTAR_REQUIRED')
    if not members or any(Path(n).is_absolute() or '..' in Path(n).parts for n in members):
        raise ValueError('SOURCE_RUNTIME_MEMBER_NAMES_INVALID')
    output.mkdir(parents=True, exist_ok=False)
    command = [tool, '--fast-read', '-xf', str(archive.resolve()), '-C', str(output.resolve()), '--', *members]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('SOURCE_RUNTIME_MEMBER_READ_FAILED:' + result.stderr.strip())
    for name in members:
        path = output/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('SOURCE_RUNTIME_REGULAR_MEMBER_MISSING:' + name)
    return command


def recover_runtime(transfer_archive: Path, reference_archive: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_RUNTIME_MEMBER_INTAKE', original_archive_complete=False,
                  source_runtime_executed=False, source_fidelity_qualified=False,
                  production_episodes=0, started_utc=datetime.now(timezone.utc).isoformat())
    try:
        transfer, _ = bound_json(transfer_archive, 'curl_response/source.json')
        prefix = delivered_file(transfer_archive, 'curl_response/lgnmodel.zip.part')
        if (transfer['mode'] != 'single_curl_response_no_resume'
                or transfer['status'] != 'FAILED' or transfer['bytes'] != prefix.stat().st_size
                or not 0 < transfer['bytes'] < transfer['expected_bytes']):
            raise ValueError('SOURCE_RUNTIME_SINGLE_INCOMPLETE_RESPONSE_REQUIRED')
        report.update(source_response_sha256=sha256_file(prefix),
                      source_response_bytes=prefix.stat().st_size,
                      expected_archive_bytes=transfer['expected_bytes'],
                      source_response_receipt_sha256=sha256_file(transfer_archive/'curl_response/source.json'))
        report['command'] = extract_complete_members(prefix, output/'lgnmodel', RUNTIME_MEMBERS)
        report['tool_version'] = subprocess.run([report['command'][0], '--version'],
            check=True, capture_output=True, text=True).stdout.strip()
        report['members'] = [dict(path=name, bytes=(output/'lgnmodel'/name).stat().st_size,
            sha256=sha256_file(output/'lgnmodel'/name)) for name in RUNTIME_MEMBERS]
        report['empty_source_members'] = [row['path'] for row in report['members'] if row['bytes'] == 0]
        comparisons = {}
        for name, reference in [('__init__.py', 'lgnmodel_init/source_asset.txt'),
                                ('cellmodel.py', 'lgnmodel_cellmodel/source_asset.txt')]:
            expected = delivered_file(reference_archive, reference)
            comparisons[name] = dict(reference_sha256=sha256_file(expected),
                extracted_sha256=sha256_file(output/'lgnmodel'/name))
            comparisons[name]['exact'] = (comparisons[name]['reference_sha256'] == comparisons[name]['extracted_sha256'])
        report['independent_source_file_comparisons'] = comparisons
        report['all_reference_files_exact'] = all(row['exact'] for row in comparisons.values())
        report['status'] = ('SOURCE_RUNTIME_MEMBERS_ACQUIRED' if report['all_reference_files_exact']
                            and not report['empty_source_members'] else 'SOURCE_RUNTIME_MEMBERS_INCOMPLETE_OR_DIFFERENT')
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        freeze_json(output/'intake.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('extract')
    for name in ('archive', 'inventory-archive', 'output'):
        extract.add_argument('--'+name, type=Path, required=True)
    inspect = commands.add_parser('inspect-metadata')
    for name in ('tables-archive', 'legacy-archive', 'current-archive', 'output'):
        inspect.add_argument('--'+name, type=Path, required=True)
    runtime = commands.add_parser('recover-runtime')
    for name in ('transfer-archive', 'reference-archive', 'output'):
        runtime.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        extract_tables(args.archive, args.inventory_archive, args.output)
    elif args.command == 'recover-runtime':
        report = recover_runtime(args.transfer_archive, args.reference_archive, args.output)
        if report['status'] != 'SOURCE_RUNTIME_MEMBERS_ACQUIRED':
            raise SystemExit(2)
    else:
        report = inspect_metadata(args.tables_archive, args.legacy_archive, args.current_archive, args.output)
        if report['status'] != 'SOURCE_LGN_TABLE_AND_LEGACY_METADATA_EXACT':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
