"""Reuse completed CA3 analysis phases from a bound resource-blocked delivery.

Only the normalized checkpoint and its source are restaged. The complete original
input identities and frozen implementation remain bound through the prior copy
receipt; the missing independent bytewise normalization replay still must run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


INPUT_IDENTITY_FIELDS = ('role', 'source', 'relative', 'bytes', 'sha256', 'delivery_sha256')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _rows(delivery: dict) -> dict[str, dict]:
    rows = {row['path']: row for row in delivery['files']}
    if len(rows) != len(delivery['files']):
        raise ValueError('CA3_REUSE_DUPLICATE_DELIVERY_PATH')
    for name in rows:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('CA3_REUSE_INVALID_DELIVERY_PATH')
    return rows


def _bound_json(archive: Path, rows: dict, name: str) -> dict:
    path, row = archive / name, rows[name]
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']
            or _sha(path) != row['sha256']):
        raise ValueError('CA3_REUSE_BOUND_METADATA_DIFFERS:' + name)
    return json.loads(path.read_text())


def prepare_reuse(archive: Path, delivery_sha256: str, work: Path,
                  planned: list[dict], prefix: Path, formation: Path,
                  native_reference: bool) -> tuple[list[dict], dict]:
    """Validate prior evidence and return only the inputs needed by replay.

    No simulation, normalization, or bulk historical rehash occurs here. Binary
    rows returned by this function undergo the worker's normal verified copying
    before replay; the prior copied-files receipt binds unstaged original inputs.
    """
    archive, work = Path(archive), Path(work)
    delivery_path = archive / 'delivery.json'
    if (len(delivery_sha256) != 64 or delivery_path.is_symlink()
            or _sha(delivery_path) != delivery_sha256):
        raise ValueError('CA3_REUSE_DELIVERY_HASH_DIFFERS')
    delivery = json.loads(delivery_path.read_text())
    units = 84053 if native_reference else 168106
    if (delivery['status'] != 'RESOURCE_BLOCKED' or delivery['exit_code'] != 3
            or delivery['copied_files_verified'] is not True
            or delivery['resource_guard_status'] != 'RESOURCE_BLOCKED'
            or delivery['native_reference'] is not native_reference
            or delivery['expected_core_neurons'] != units
            or delivery['prefix_archive'] != str(prefix)
            or delivery['formation_archive'] != str(formation)):
        raise ValueError('CA3_REUSE_BOUND_RESOURCE_BLOCKED_ANALYSIS_REQUIRED')
    rows = _rows(delivery)
    analysis = _bound_json(archive, rows, 'analysis.json')
    guard = _bound_json(archive, rows, 'phases/analysis/receipt.json')
    if (guard['status'] != 'RESOURCE_BLOCKED' or guard['command_started'] is not True
            or str(guard['job_id']) != str(delivery['job_id'])
            or analysis['status'] != 'PREPARING_INPUTS'
            or str(analysis['job_id']) != str(delivery['job_id'])
            or analysis['native_reference'] is not native_reference
            or analysis['expected_core_neurons'] != units):
        raise ValueError('CA3_REUSE_INTERRUPTED_ANALYSIS_REQUIRED')
    phases = analysis['phases']
    if ([phase['name'] for phase in phases] != ['formation_replay', 'normalization']
            or any(phase['exit_code'] != 0 for phase in phases)
            or 'normalization_replay.json' in rows):
        raise ValueError('CA3_REUSE_TWO_COMPLETE_PHASES_AND_MISSING_REPLAY_REQUIRED')
    for phase in phases:
        if phase['command'][1:4] != ['-B', '-m', 'population_dynamics.carlsim_' + phase['name']]:
            raise ValueError('CA3_REUSE_PHASE_COMMAND_DIFFERS')
    identity = lambda values: [{key: row[key] for key in INPUT_IDENTITY_FIELDS} for row in values]
    if identity(analysis['inputs']) != identity(planned):
        raise ValueError('CA3_REUSE_ORIGINAL_INPUT_IDENTITIES_DIFFER')
    if analysis['input_bytes'] != sum(row['bytes'] for row in planned):
        raise ValueError('CA3_REUSE_ORIGINAL_INPUT_BYTES_DIFFER')
    inputs = {(row['role'], row['relative']): row for row in planned}
    if len(inputs) != len(planned):
        raise ValueError('CA3_REUSE_DUPLICATE_INPUT_IDENTITY')

    # These are current staged copies, checked against the original frozen receipt.
    # Include every original Python module, rather than only direct replay imports.
    code_rows = {name: row for name, row in rows.items()
                 if name.startswith('Libs/') and name.endswith('.py')}
    required_modules = ['carlsim_formation_replay', 'carlsim_normalization',
                        'carlsim_normalization_replay', 'carlsim_records',
                        'carlsim_prefix_replay', 'carlsim_event_order', 'provenance',
                        'resource_probe']
    for name in required_modules:
        if 'Libs/population_dynamics/' + name + '.py' not in code_rows:
            raise ValueError('CA3_REUSE_FROZEN_MODULE_MISSING:' + name)
    for name, row in {**code_rows, 'run_guarded.py': rows['run_guarded.py']}.items():
        current = work / name
        if (current.is_symlink() or current.stat().st_size != row['bytes']
                or _sha(current) != row['sha256']):
            raise ValueError('CA3_REUSE_FROZEN_SOFTWARE_DIFFERS:' + name)
    for name, key in [('software_delivery.json', 'software_delivery_sha256'),
                      ('software_manifest.json', 'software_manifest_sha256'),
                      ('normalization_software_delivery.json', 'normalization_software_delivery_sha256')]:
        _bound_json(archive, rows, name)
        if rows[name]['sha256'] != delivery[key] or _sha(work / name) != delivery[key]:
            raise ValueError('CA3_REUSE_SOFTWARE_RECEIPT_DIFFERS:' + name)

    replay = _bound_json(archive, rows, 'formation_replay.json')
    normalization = _bound_json(archive, rows, 'normalization.json')
    wanted = ('PASS_NATIVE_FORMATION_PREFIX_REPLAY_AND_COUNTS' if native_reference
              else 'PASS_COUPLED_EXTENSION_FORMATION_PREFIX_REPLAY_AND_COUNTS')
    if replay['status'] != wanted or replay['units'] != units:
        raise ValueError('CA3_REUSE_FORMATION_REPLAY_STATUS_DIFFERS')
    if replay['implementation_sha256'] != code_rows['Libs/population_dynamics/carlsim_formation_replay.py']['sha256']:
        raise ValueError('CA3_REUSE_FORMATION_IMPLEMENTATION_DIFFERS')
    helpers = {name: code_rows['Libs/population_dynamics/' + name + '.py']['sha256']
               for name in ['carlsim_records', 'carlsim_prefix_replay', 'carlsim_event_order']}
    if replay['helper_implementation_sha256'] != helpers:
        raise ValueError('CA3_REUSE_FORMATION_HELPERS_DIFFER')
    for role, field in [('prefix', 'original'), ('formation', 'formation')]:
        for relative, suffix in [('native_report.json', 'report_sha256'),
                                 ('observations/recording.json', 'recording_sha256')]:
            if replay[field + '_' + suffix] != inputs[(role, 'inputs/' + role + '/' + relative)]['sha256']:
                raise ValueError('CA3_REUSE_FORMATION_SOURCE_REPORT_DIFFERS')
    source = inputs[('formation', 'inputs/formation/native_run/network65p.dat')]
    normalized_name = 'network65p_assem275_divnorm.dat' if native_reference else 'network65p_assem550_divnorm.dat'
    normalized = rows[normalized_name]
    if (normalization['status'] != 'APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION'
            or normalization['source_units'] != units
            or normalization['source_sha256'] != source['sha256']
            or normalization['output_sha256'] != normalized['sha256']
            or normalization['output_bytes'] != normalized['bytes']
            or normalization['implementation_sha256'] != code_rows['Libs/population_dynamics/carlsim_normalization.py']['sha256']):
        raise ValueError('CA3_REUSE_NORMALIZATION_BINDING_DIFFERS')

    selected = [dict(source)]
    for name in [normalized_name, 'formation_replay.json', 'normalization.json']:
        row = rows[name]
        selected.append(dict(role='reused_analysis', source=str(archive / name),
                             relative='analysis/' + name, bytes=row['bytes'],
                             sha256=row['sha256'], delivery_sha256=delivery_sha256))
    provenance = dict(archive=str(archive), job_id=delivery['job_id'],
                      delivery_sha256=delivery_sha256, original_status=delivery['status'],
                      original_exit_code=delivery['exit_code'],
                      analysis_sha256=rows['analysis.json']['sha256'],
                      guard_receipt_sha256=rows['phases/analysis/receipt.json']['sha256'],
                      original_inputs=analysis['inputs'], original_input_bytes=analysis['input_bytes'],
                      original_phases=phases,
                      original_phase_seconds=sum(phase['elapsed_seconds'] for phase in phases),
                      frozen_module_hashes={name: row['sha256'] for name, row in code_rows.items()},
                      reused_report_hashes={name: rows[name]['sha256']
                                            for name in ['formation_replay.json', 'normalization.json']},
                      original_binary_inputs_rehashed=False,
                      prior_verified_input_copies_reused=True,
                      new_numerical_phase='normalization_replay')
    return selected, provenance
