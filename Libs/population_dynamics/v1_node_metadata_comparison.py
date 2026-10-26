"""Compare all legacy/current V1 nodes and bind existing weight witnesses."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_graph_identity import _types
from .v1_sonata_witness import RangeFile, wall_deadline

IDENTIFIERS = ('node_id', 'node_group_id', 'node_group_index', 'node_type_id')
PROPERTIES = ('x', 'y', 'z', 'tuning_angle')
WITNESSES = (577, 162949, 26879, 26846, 220371, 138674)
MAX_BYTES = 16 * 1024 * 1024
MAX_SECONDS = 180
RANGE_BYTES = 512 * 1024
RECOVERY_BODY_LIMIT = 32 * 1024 * 1024
MAX_RECOVERY_ATTEMPTS = 4


def metadata(path):
    """Preserve missing properties; use source group indices rather than row order."""
    values, schema = {}, {}
    with h5py.File(path, 'r') as handle:
        group = handle['nodes/v1']
        for name in IDENTIFIERS:
            if name not in group:
                raise ValueError('V1_NODE_ID_MAPPING_FIELD_MISSING:' + name)
            dataset = group[name]
            if dataset.ndim != 1 or dataset.dtype.kind not in 'iu':
                raise ValueError('V1_NODE_IDENTIFIER_SCHEMA_INVALID:' + name)
            values[name] = dataset[...]
            schema[name] = dict(shape=list(dataset.shape), dtype=dataset.dtype.str)
        n = len(values['node_id'])
        if not n or any(len(values[k]) != n for k in IDENTIFIERS):
            raise ValueError('V1_NODE_IDENTIFIER_DENOMINATOR_MISMATCH')
        if any(np.any(values[k] < 0) for k in IDENTIFIERS):
            raise ValueError('V1_NODE_NEGATIVE_IDENTIFIER')
        if len(np.unique(values['node_id'])) != n:
            raise ValueError('V1_DUPLICATE_NODE_ID')
        for name in PROPERTIES:
            data = np.full(n, np.nan, dtype=np.float64)
            present = np.zeros(n, dtype=bool)
            dtypes = np.full(n, '', dtype='U8')
            groups = {}
            for group_id in np.unique(values['node_group_id']):
                key = str(int(group_id))
                selected = np.flatnonzero(values['node_group_id'] == group_id)
                indices = values['node_group_index'][selected]
                if key not in group or name not in group[key]:
                    groups[key] = dict(status='MISSING', affected_nodes=len(selected))
                    continue
                dataset = group[key][name]
                if (dataset.ndim != 1 or dataset.dtype.kind not in 'fiu'
                        or (len(indices) and int(indices.max()) >= len(dataset))):
                    raise ValueError('V1_NODE_PROPERTY_SCHEMA_OR_INDEX_INVALID:' + key + '/' + name)
                # Float64 exactly represents these stored coordinate/tuning floats.
                if dataset.dtype.kind in 'iu' and dataset.dtype.itemsize > 4:
                    raise ValueError('V1_INTEGER_PROPERTY_REQUIRES_EXACT_REPRESENTATION:' + name)
                data[selected] = dataset[...][indices]
                present[selected] = True
                dtypes[selected] = dataset.dtype.str
                groups[key] = dict(status='PRESENT', shape=list(dataset.shape), dtype=dataset.dtype.str)
            values[name] = data
            values[name + '_present'] = present
            values[name + '_dtype'] = dtypes
            schema[name] = dict(groups=groups, present_nodes=int(present.sum()), missing_nodes=int((~present).sum()))
    return values, schema


def scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float):
        return dict(value=value if np.isfinite(value) else None, repr=repr(value), hex=value.hex())
    return value


def compare_metadata(legacy_path, current_path, legacy_types, current_types, witnesses=()):
    legacy, legacy_schema = metadata(legacy_path)
    current, current_schema = metadata(current_path)
    old_order, new_order = np.argsort(legacy['node_id']), np.argsort(current['node_id'])
    old_ids, new_ids = legacy['node_id'][old_order], current['node_id'][new_order]
    if not np.array_equal(old_ids, new_ids):
        raise ValueError('V1_NODE_ID_SETS_NOT_COMPARABLE')
    n = len(old_ids)
    result = dict(nodes=n, id_mapping=dict(unique_ids=True, same_id_set=True,
        row_order_equal=bool(np.array_equal(legacy['node_id'], current['node_id'])),
        comparison_order='ascending node_id', property_mapping='node_group_id/node_group_index'),
        schemas=dict(legacy=legacy_schema, current=current_schema), fields={}, witnesses=[])
    for name in IDENTIFIERS + PROPERTIES:
        a, b = legacy[name][old_order], current[name][new_order]
        if name in IDENTIFIERS:
            both = np.ones(n, dtype=bool)
            equal = a == b
            detail = dict(legacy_dtype=legacy_schema[name]['dtype'], current_dtype=current_schema[name]['dtype'])
        else:
            ap, bp = legacy[name + '_present'][old_order], current[name + '_present'][new_order]
            both = ap & bp
            equal = (a == b) | (np.isnan(a) & np.isnan(b))
            detail = dict(legacy_missing=int((~ap).sum()), current_missing=int((~bp).sum()),
                presence_mismatches=int((ap != bp).sum()),
                legacy_nonfinite=int((ap & ~np.isfinite(a)).sum()), current_nonfinite=int((bp & ~np.isfinite(b)).sum()),
                both_nan=int((both & np.isnan(a) & np.isnan(b)).sum()),
                signed_zero_mismatches=int((both & (a == 0) & (b == 0) & (np.signbit(a) != np.signbit(b))).sum()),
                storage_dtype_mismatches=int((both & (legacy[name+'_dtype'][old_order] != current[name+'_dtype'][new_order])).sum()))
            finite = both & np.isfinite(a) & np.isfinite(b)
            detail['finite_max_absolute_difference'] = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None
        different = both & ~equal
        detail.update(compared_nodes=int(both.sum()), equal_nodes=int((both & equal).sum()),
                      different_nodes=int(different.sum()),
                      difference_examples=[dict(node_id=int(old_ids[i]), legacy=scalar(a[i]), current=scalar(b[i]))
                                           for i in np.flatnonzero(different)[:8]])
        result['fields'][name] = detail
    old_type_ids = legacy['node_type_id'][old_order]
    new_type_ids = current['node_type_id'][new_order]
    for ids, table in ((old_type_ids, legacy_types), (new_type_ids, current_types)):
        if not set(map(int, np.unique(ids))) <= table.keys():
            raise ValueError('V1_NODE_TYPE_DECLARATION_MISSING')
    columns = sorted(set().union(*(set(row) for row in list(legacy_types.values()) + list(current_types.values()))))
    transitions, counts = np.unique(np.column_stack((old_type_ids, new_type_ids)), axis=0, return_counts=True)
    result['node_type_transitions'] = []
    semantic_differences = dict.fromkeys(columns, 0)
    for (old, new), count in zip(transitions, counts):
        old, new, count = int(old), int(new), int(count)
        changes = {key: dict(legacy=legacy_types[old].get(key), current=current_types[new].get(key))
                   for key in columns if legacy_types[old].get(key) != current_types[new].get(key)}
        for key in changes:
            semantic_differences[key] += count
        result['node_type_transitions'].append(dict(legacy_node_type_id=old, current_node_type_id=new, nodes=count,
                                                   declaration_changes=changes))
    result['node_declaration_difference_counts'] = semantic_differences
    common = sorted(legacy_types.keys() & current_types.keys())
    result['type_tables'] = dict(legacy_rows=len(legacy_types), current_rows=len(current_types), common_type_ids=len(common),
        legacy_only_type_ids=sorted(legacy_types.keys()-current_types.keys()),
        current_only_type_ids=sorted(current_types.keys()-legacy_types.keys()),
        same_id_declaration_differences=[dict(node_type_id=k, fields={
            c: dict(legacy=legacy_types[k].get(c), current=current_types[k].get(c))
            for c in columns if legacy_types[k].get(c) != current_types[k].get(c)})
            for k in common if legacy_types[k] != current_types[k]])
    for node in witnesses:
        matches = np.flatnonzero(old_ids == node)
        if len(matches) != 1:
            raise ValueError('V1_WITNESS_NODE_NOT_PRESENT:' + str(node))
        i = int(matches[0])
        row = dict(node_id=node)
        for label, data, order, table in (('legacy', legacy, old_order, legacy_types), ('current', current, new_order, current_types)):
            index = order[i]
            row[label] = {name: scalar(data[name][index]) if name in IDENTIFIERS or data[name+'_present'][index]
                          else dict(status='MISSING') for name in IDENTIFIERS + PROPERTIES}
            row[label]['type_declaration'] = table[int(data['node_type_id'][index])]
        result['witnesses'].append(row)
    return result


def bind_witnesses(result, arithmetic, sonata, association):
    rows = {row['node_id']: row for row in result['witnesses']}
    old_cases = {row['name']: row for row in arithmetic['cases']}
    sonata_cases = {row['name']: row for row in sonata['cases']}
    result['weight_witness_binding'] = []
    for case in association['cases']:
        old, source = old_cases[case['name']], sonata_cases[case['name']]
        if any(case[k] != old[k] or case[k] != source[k] for k in ('source_id', 'target_id')):
            raise ValueError('V1_WITNESS_PAIR_IDENTITY_MISMATCH')
        if not source['matches_complete'] or len(case['source_rows']) != len(source['matching_rows']):
            raise ValueError('V1_WITNESS_SOURCE_ROWS_INCOMPLETE')
        target, source_node = rows[case['target_id']], rows[case['source_id']]
        if target['current']['node_type_id'] != old['target_node_type_id']:
            raise ValueError('V1_CURRENT_TARGET_ARITHMETIC_BINDING_DIFFERS')
        for edge in case['source_rows']:
            matching = [v for v in source['matching_rows'] if v['global_row_id'] == edge['global_row_id']]
            if len(matching) != 1 or matching[0]['edge_type_id'] != edge['edge_type_id']:
                raise ValueError('V1_SONATA_EDGE_TYPE_BINDING_DIFFERS')
            target_match = re.fullmatch(r"node_type_id=='([0-9]+)'", edge['table_row']['target_query'])
            population_match = re.fullmatch(r"pop_name=='([^']+)'", edge['table_row']['source_query'])
            if target_match is None or population_match is None:
                raise ValueError('V1_UNSUPPORTED_SOURCE_QUERY')
            expected = int(target_match[1])
            result['weight_witness_binding'].append(dict(name=case['name'], source_id=case['source_id'], target_id=case['target_id'],
                global_edge_row=edge['global_row_id'], edge_type_id=edge['edge_type_id'], nsyns=edge['nsyns'],
                source_table_target_type_id=expected,
                legacy_target_type_id=target['legacy']['node_type_id'], current_target_type_id=target['current']['node_type_id'],
                legacy_target_predicate_matches=target['legacy']['node_type_id'] == expected,
                current_target_predicate_matches=target['current']['node_type_id'] == expected,
                legacy_source_population_matches=source_node['legacy']['type_declaration']['pop_name'] == population_match[1],
                current_source_population_matches=source_node['current']['type_declaration']['pop_name'] == population_match[1],
                existing_weight_exact=edge['exact'], existing_weight_absolute_residual=edge['absolute_residual']))


def download_legacy(source, proxy_url, output, report, resume=None, range_bytes=RANGE_BYTES):
    """Retain only fully checked ranges, following existing source-prefix semantics."""
    if type(range_bytes) is not int or not 0 < range_bytes <= RANGE_BYTES:
        raise ValueError('V1_METADATA_RANGE_LIMIT_EXCEEDED')
    identity = {key: source[key] for key in ('etag', 'expected_bytes', 'parent_index_sha256')}
    report['source_identity'] = identity
    previous_body, attempt, verified = 0, 1, 0
    digest = hashlib.sha256()
    partial = output/'legacy_v1_nodes.h5.part'
    reader = None
    try:
        with partial.open('xb') as stream:
            if resume is not None:
                prior = json.loads((resume/'report.json').read_text())
                asset = prior['legacy_asset']
                # A matching numeric prefix length alone cannot establish source identity.
                if (prior.get('source_identity') != identity or
                        asset['path'] not in ('legacy_v1_nodes.h5.part', 'legacy_v1_nodes.h5')):
                    raise ValueError('V1_RESUME_SOURCE_IDENTITY_DIFFERS')
                prefix = resume/asset['path']
                count = prior.get('verified_prefix_bytes')
                expected_hash = prior.get('verified_prefix_sha256')
                if (type(count) is not int or not 0 <= count <= source['expected_bytes']
                        or prefix.stat().st_size != count or asset['bytes'] != count
                        or not expected_hash or sha256_file(prefix) != expected_hash
                        or asset['sha256'] != expected_hash):
                    raise ValueError('V1_VERIFIED_METADATA_PREFIX_REQUIRED')
                previous_body = prior['recovery']['response_body_upper_bound_bytes']
                attempt = prior['recovery']['attempt'] + 1
                if (type(previous_body) is not int or previous_body < 0 or
                        previous_body >= RECOVERY_BODY_LIMIT or attempt > MAX_RECOVERY_ATTEMPTS):
                    raise ValueError('V1_NODE_RECOVERY_BUDGET_EXHAUSTED')
                with prefix.open('rb') as old:
                    while verified < count:
                        block = old.read(min(1024 * 1024, count-verified))
                        if not block:
                            raise ValueError('V1_RESUME_PREFIX_TRUNCATED')
                        stream.write(block);digest.update(block);verified += len(block)
                report.update(resumed_prefix_bytes=verified, resumed_report_sha256=sha256_file(resume/'report.json'))
            if verified < source['expected_bytes']:
                reader = RangeFile(source, proxy_url, max_bytes=min(MAX_BYTES,RECOVERY_BODY_LIMIT-previous_body),
                                   max_seconds=MAX_SECONDS)
                reader.seek(verified)
                with wall_deadline(MAX_SECONDS):
                    while verified < source['expected_bytes']:
                        length = min(range_bytes,source['expected_bytes']-verified)
                        block = reader.read(length)
                        if len(block) != length:
                            raise ValueError('V1_COMPLETE_NODE_RANGE_LENGTH_MISMATCH')
                        # RangeFile validates TLS/ETag/range/length before returning.
                        stream.write(block);stream.flush();digest.update(block);verified += len(block)
        if verified == source['expected_bytes']:
            destination = output/'legacy_v1_nodes.h5'
            partial.rename(destination)
            return destination
    finally:
        counters = reader.counters() if reader is not None else dict(response_body_bytes=0,requested_bytes=0,redirect_body_bytes=0)
        # A failed low-level read may have consumed part of its final block.
        # Charge the entire requested range as an upper bound for the total cap.
        upper = counters['response_body_bytes'] if reader is None or reader.failed is None else max(
            counters['response_body_bytes'],counters['requested_bytes']+counters['redirect_body_bytes'])
        report['http'] = counters
        report['recovery'] = dict(attempt=attempt,max_attempts=MAX_RECOVERY_ATTEMPTS,
            range_bytes=range_bytes,attempt_body_upper_bound_bytes=upper,
            response_body_upper_bound_bytes=previous_body+upper,maximum_response_body_bytes=RECOVERY_BODY_LIMIT)
        if reader is not None:
            reader.close()
        destination = output/'legacy_v1_nodes.h5'
        artifact = destination if destination.exists() else partial
        report.update(verified_prefix_bytes=verified, verified_prefix_sha256=digest.hexdigest(),
            legacy_asset=dict(path=artifact.name,bytes=verified,sha256=digest.hexdigest(),
                              etag=source['etag'],complete=destination.exists()))


def run(inputs, proxy_url, output, resume=None, range_bytes=RANGE_BYTES):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='FAILED_NODE_METADATA_COMPARISON', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), scientific_qualification=False,
        source_qualification=False, production_episodes=0, full_graph_recompared=False,
        scope='Complete legacy/current node metadata comparison and binding of three existing weight witnesses.',
        numeric_comparison='Exact value equality; paired NaNs, signed zero and storage dtypes reported separately.')
    try:
        names = ('legacy_source.json','legacy_types.json','current_types.json','node_validation.json',
                 'arithmetic.json','sonata.json','association.json')
        documents = {name: json.loads((inputs/name).read_text()) for name in names}
        source, validation = documents['legacy_source.json'], documents['node_validation.json']
        report['input_sha256'] = {p.name: sha256_file(p) for p in [*(inputs/n for n in names),
            inputs/'legacy_types.txt',inputs/'current_types.txt',inputs/'current_nodes.h5']}
        hashes = report['input_sha256']
        if (validation['status'] != 'NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS'
                or hashes['current_nodes.h5'] != validation['nodes_sha256']
                or hashes['current_types.txt'] != validation['node_types_sha256']
                or hashes['current_types.txt'] != documents['current_types.json']['sha256']
                or hashes['legacy_types.txt'] != documents['legacy_types.json']['sha256']
                or source['parent_index_sha256'] != documents['legacy_types.json']['parent_index_sha256']):
            raise ValueError('V1_NODE_METADATA_SOURCE_BINDING_DIFFERS')
        arithmetic, sonata, association = (documents[n+'.json'] for n in ('arithmetic','sonata','association'))
        if (association['arithmetic_report_sha256'] != hashes['arithmetic.json']
                or association['sonata_report_sha256'] != hashes['sonata.json']
                or sonata['witness_report_sha256'] != hashes['arithmetic.json']):
            raise ValueError('V1_EXISTING_WITNESS_HASH_CHAIN_DIFFERS')
        if source['expected_bytes'] > MAX_BYTES:
            raise ValueError('V1_LEGACY_NODE_ASSET_EXCEEDS_METADATA_BUDGET')
        destination = download_legacy(source,proxy_url,output,report,resume,range_bytes)
        result = compare_metadata(destination, inputs/'current_nodes.h5', _types(inputs/'legacy_types.txt'),
                                  _types(inputs/'current_types.txt'), WITNESSES)
        if result['nodes'] != validation['cortical_units']:
            raise ValueError('V1_FULL_CORTICAL_DENOMINATOR_DIFFERS')
        bind_witnesses(result, arithmetic, sonata, association)
        transitions = result.pop('node_type_transitions')
        freeze_json(output/'node_type_transitions.json', transitions)
        report.update(status='COMPLETE_NODE_METADATA_COMPARISON', exit_code=0, comparison=result,
            node_type_transitions=dict(path='node_type_transitions.json', rows=len(transitions),
                                      sha256=sha256_file(output/'node_type_transitions.json')))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            runtime=runtime_manifest(dict(max_bytes=MAX_BYTES,max_seconds=MAX_SECONDS,range_bytes=range_bytes),list(sys.argv),report['exit_code']))
        freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--proxy-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--range-bytes', type=int, default=RANGE_BYTES)
    args = parser.parse_args()
    return run(**vars(args))['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
