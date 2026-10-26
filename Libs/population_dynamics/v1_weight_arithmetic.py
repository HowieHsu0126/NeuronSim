"""Bounded arithmetic witnesses for a completed source-weight comparison."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from .provenance import freeze_json, runtime_manifest, sha256_file
from . import v1_synaptic_weights as weights


def point_candidates(rule, row, metadata, source_ids, target_ids, candidates):
    source = pd.DataFrame({k: metadata[k][source_ids] for k in ('x', 'z', 'tuning_angle')})
    target = pd.DataFrame({k: metadata[k][target_ids] for k in ('x', 'z', 'tuning_angle')})
    edges = pd.DataFrame({'syn_weight': np.full(len(source_ids), row['syn_weight']),
                          'weight_sigma': np.full(len(source_ids), row['weight_sigma']),
                          'nsyns': np.zeros(len(source_ids), dtype=np.int64)})
    results = []
    for number in candidates:
        edges['nsyns'] = number
        value = np.asarray(rule(edges, source, target))
        if value.shape != source_ids.shape or not np.isfinite(value).all():
            raise ValueError('DIAGNOSTIC_SOURCE_FUNCTION_OUTPUT_INVALID')
        results.append(value)
    return np.stack(results)


def select_case(case, raw, metadata, types, sources, batch_size):
    """Keep the original batch boundaries and within-pair ordering for arithmetic."""
    mode = case['selection']
    if mode not in ('raw_location', 'first', 'max_absolute_residual'):
        raise ValueError('DIAGNOSTIC_SELECTION_INVALID')
    if mode == 'raw_location':
        group = raw['edges'][case['raw_group']]
        source_id, target_id = int(group['source'][case['raw_offset']]), int(group['target'][case['raw_offset']])
        key = (types[int(metadata['node_type_id'][source_id])]['pop_name'],
               int(metadata['node_type_id'][target_id]))
        if source_id != case['expected_source_id'] or target_id != case['expected_target_id']:
            raise ValueError('DIAGNOSTIC_RAW_LOCATION_IDENTITY_DIFFERS')
    else:
        key = (case['source_population'], case['target_node_type_id'])
    row = sources['table'][key]
    target_population = types[key[1]]['pop_name']
    candidates = weights.allowed_multiplicities(sources['compute'], key[0], target_population)
    source_type_ids = np.asarray([kind for kind, item in types.items() if item['pop_name'] == key[0]])
    selected = None
    for group_index, group in enumerate(raw['edges']):
        if mode == 'raw_location' and group_index != case['raw_group']:
            continue
        source_ids, target_ids = np.asarray(group['source']), np.asarray(group['target'])
        matches = np.flatnonzero(np.isin(metadata['node_type_id'][source_ids], source_type_ids)
                                 & (metadata['node_type_id'][target_ids] == key[1]))
        if mode == 'raw_location':
            starts = [case['raw_offset'] // batch_size * batch_size]
        elif not len(matches):
            continue
        elif mode == 'first':
            starts = [int(matches[0]) // batch_size * batch_size]
        else:
            starts = np.unique(matches // batch_size) * batch_size
        for start in starts:
            indices = matches[(matches >= start) & (matches < start + batch_size)]
            if not len(indices):
                continue
            src, dst = source_ids[indices], target_ids[indices]
            actual = np.asarray(group['params']['weight'])[indices]
            predicted = point_candidates(sources['rules'][row['weight_function']], row,
                                         metadata, src, dst, candidates)
            stored = predicted.astype(actual.dtype)
            residual = np.min(np.abs(stored.astype(np.float64) - actual.astype(np.float64)), axis=0)
            if mode == 'raw_location':
                local = int(np.flatnonzero(indices == case['raw_offset'])[0])
            elif mode == 'first':
                local = 0
            else:
                local = int(np.argmax(residual))
            if selected is None or float(residual[local]) > selected['nearest_absolute_residual']:
                selected = dict(key=key, row=row, candidates=candidates, raw_group=group_index,
                    raw_offset=int(indices[local]), batch_start=int(start), pair_batch_tuples=len(indices),
                    source_id=int(src[local]), target_id=int(dst[local]), source_ids=src, target_ids=dst,
                    local_index=local, raw_weight=actual[local], point_default=predicted[:, local],
                    nearest_absolute_residual=float(residual[local]))
            if mode != 'max_absolute_residual':
                return selected
    if selected is None:
        raise ValueError('DIAGNOSTIC_SELECTOR_HAS_NO_TUPLES:' + case['name'])
    return selected


def numeric_record(value):
    number = float(value)
    return dict(value=number, repr=repr(number), hex=number.hex(), scalar_type=type(value).__name__)


def trace_case(case, selected, raw, metadata, literal_row, bio_rules, point_rules, nodes_path):
    row, candidates = selected['row'], selected['candidates']
    source_id, target_id = selected['source_id'], selected['target_id']
    actual = np.asarray(selected['raw_weight'])
    group = raw['edges'][selected['raw_group']]
    delay = np.asarray(group['params']['delay'])
    delay = delay.item() if delay.ndim == 0 else delay[selected['raw_offset']]
    trace = dict(name=case['name'], selector=case, raw_group=selected['raw_group'],
        raw_offset=selected['raw_offset'], source_id=source_id, target_id=target_id,
        raw_weight=numeric_record(actual.item()), raw_weight_dtype=actual.dtype.str,
        raw_delay=numeric_record(delay), raw_receptor=int(group['params']['receptor_type']),
        source_population=selected['key'][0], target_node_type_id=selected['key'][1],
        table_row=row, numeric_tokens={k:literal_row[k] for k in ('syn_weight','weight_sigma','delay')},
        allowed_nsyns=list(candidates), original_batch_start=selected['batch_start'],
        original_pair_batch_tuples=selected['pair_batch_tuples'],
        main_path_nearest_absolute_residual=selected['nearest_absolute_residual'],
        source_node={}, target_node={}, variants=[])
    source_node = {k:metadata[k][source_id] for k in ('x','z','tuning_angle')}
    target_node = {k:metadata[k][target_id] for k in ('x','z','tuning_angle')}
    with h5py.File(nodes_path) as handle:
        native = handle['nodes/v1']
        for name, identity, values in (('source_node',source_id,source_node),('target_node',target_id,target_node)):
            trace[name] = {k:dict(numeric_record(v),dtype=metadata[k].dtype.str) for k,v in values.items()}
            trace[name].update(node_group_id=int(native['node_group_id'][identity]),
                               node_group_index=int(native['node_group_index'][identity]))
    for parsing in ('pandas_default','literal_float'):
        used_row = dict(row)
        if parsing == 'literal_float':
            used_row.update({k:float(literal_row[k]) for k in ('syn_weight','weight_sigma','delay')})
        rule = row['weight_function']
        batch = (selected['point_default'] if parsing == 'pandas_default' else
                 point_candidates(point_rules[rule],used_row,metadata,selected['source_ids'],
                                  selected['target_ids'],candidates)[:,selected['local_index']])
        single = point_candidates(point_rules[rule],used_row,metadata,
                                  np.asarray([source_id]),np.asarray([target_id]),candidates)[:,0]
        per_synapse = bio_rules[rule](used_row,source_node,target_node)
        scalar = np.asarray([per_synapse*n for n in candidates])
        for execution,values in (('pointnet_original_batch',batch),('pointnet_singleton',single),
                                 ('bionet_scalar_return_times_candidate',scalar)):
            if not np.isfinite(values).all():
                raise ValueError('DIAGNOSTIC_NONFINITE_CANDIDATE')
            stored = values.astype(actual.dtype)
            trace['variants'].append(dict(parsing=parsing,execution=execution,
                parsed_numbers={k:numeric_record(used_row[k]) for k in ('syn_weight','weight_sigma','delay')},
                candidates=[dict(nsyns=int(n),before_storage=numeric_record(v),after_storage=numeric_record(s),
                                 exact=bool(s==actual),absolute_residual=float(abs(float(s)-float(actual))))
                            for n,v,s in zip(candidates,values,stored)]))
    return trace


def diagnose(source_root: Path, graph: Path, graph_inspection: Path, node_validation: Path,
             nodes: Path, types: Path, comparison_archive: Path, config: Path, output: Path):
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='FAILED_SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC',exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(),scientific_qualification=False,
        production_episodes=0,new_full_circuits=0,original_nsyns_recovered=False,
        scope='Selected development witnesses; no tolerance, table switch or qualification change.',
        bionet_scope='Unchanged BioNet arithmetic with identical GLIF coefficients, multiplied by each candidate; not a BioNet simulation.')
    try:
        baseline_path = comparison_archive/'weights/report.json'
        delivery = json.loads((comparison_archive/'delivery.json').read_text())
        entry = next(r for r in delivery['files'] if r['path']=='weights/report.json')
        if baseline_path.stat().st_size!=entry['bytes'] or sha256_file(baseline_path)!=entry['sha256']:
            raise ValueError('DIAGNOSTIC_COMPARISON_DELIVERY_DIFFERS')
        baseline = json.loads(baseline_path.read_text())
        if baseline['status'] not in ('SOURCE_WEIGHT_RULE_CANDIDATES_EXACT_DEVELOPMENT','SOURCE_WEIGHT_RULE_DIFFERS_DIAGNOSTIC'):
            raise ValueError('DIAGNOSTIC_COMPLETE_COMPARISON_REQUIRED')
        hashes = {name:sha256_file(path) for name,path in
                  (('graph',graph),('graph_inspection',graph_inspection),('node_validation',node_validation),
                   ('nodes',nodes),('types',types))}
        if hashes != baseline['input_sha256']:
            raise ValueError('DIAGNOSTIC_BOUND_INPUT_DIFFERS_BEFORE_UNPICKLE')
        sources = weights.load_sources(source_root)
        if sources['source_hashes'] != baseline['source_sha256']:
            raise ValueError('DIAGNOSTIC_SOURCE_INPUT_DIFFERS')
        body,bio_receipt = weights.read_source(source_root/'weight_semantics/original_bionet_runner')
        bio_rules,adaptations = weights.extract_functions(body.decode(),weights.RULE_NAMES)
        assert adaptations==0
        body,_ = weights.read_source(source_root/'weight_semantics/legacy_recurrent_table')
        literal = {(weights.parse_query(r['source_query'],'pop_name'),
                    weights.parse_query(r['target_query'],'node_type_id')):r
                   for r in csv.DictReader(io.StringIO(body.decode()),delimiter=' ')}
        settings = yaml.safe_load(config.read_text())
        cases = settings['cases']
        if not isinstance(cases,list) or not 1<=len(cases)<=8 or len({r['name'] for r in cases})!=len(cases):
            raise ValueError('DIAGNOSTIC_BOUNDED_UNIQUE_CASES_REQUIRED')
        metadata,type_rows = weights._h5_metadata(nodes),weights._types(types)
        with graph.open('rb') as handle:
            raw = pickle.load(handle)
        traces=[]
        for case in cases:
            selected=select_case(case,raw,metadata,type_rows,sources,baseline['batch_size'])
            traces.append(trace_case(case,selected,raw,metadata,literal[selected['key']],
                                     bio_rules,sources['rules'],nodes))
        report.update(status='SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE',exit_code=0,
            comparison_report_sha256=entry['sha256'],input_sha256=hashes,
            source_sha256=sources['source_hashes'],bionet_source_sha256=bio_receipt['sha256'],
            configuration=settings,configuration_sha256=sha256_file(config),cases=traces)
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        report['runtime']=runtime_manifest({},list(sys.argv),report['exit_code'])
        freeze_json(output/'report.json',report)
    return report


def associate_case(witness, source_case, by_edge_type, rules):
    if (source_case.get('matches_complete') is not True
            or source_case.get('matching_row_count') != len(source_case.get('matching_rows', []))
            or any(source_case.get(key) != witness.get(key)
                   for key in ('name', 'source_id', 'target_id', 'raw_group', 'raw_offset'))):
        raise ValueError('ASSOCIATION_COMPLETE_MATCHING_WITNESS_REQUIRED')
    metadata = {key: np.asarray([witness['source_node'][key]['value'],
                                witness['target_node'][key]['value']], dtype=witness['source_node'][key]['dtype'])
                for key in ('x', 'z', 'tuning_angle')}
    actual = np.asarray(witness['raw_weight']['value'], dtype=witness['raw_weight_dtype'])
    result = dict(name=witness['name'], source_id=witness['source_id'], target_id=witness['target_id'],
                  prior_table_row=witness['table_row'], raw_weight=witness['raw_weight'], source_rows=[])
    for edge in source_case['matching_rows']:
        if any(edge[key+'_node_id'] != witness[key+'_id'] for key in ('source', 'target')):
            raise ValueError('ASSOCIATION_SOURCE_ROW_IDENTITY_INVALID')
        row = dict(by_edge_type[edge['edge_type_id']])
        properties = edge['properties']
        nsyn = properties['nsyns']
        if nsyn.get('status') != 'RECORDED' or type(nsyn.get('value')) is not int or nsyn['value'] < 1:
            raise ValueError('ASSOCIATION_RECORDED_POSITIVE_NSYNS_REQUIRED')
        effective = dict(row)
        for key in ('syn_weight', 'weight_sigma', 'delay'):
            prop = properties[key]
            if prop['status'] == 'RECORDED':
                effective[key] = prop['value']
            elif prop['status'] != 'MISSING':
                raise ValueError('ASSOCIATION_INVALID_SOURCE_PROPERTY:' + key)
        value = point_candidates(rules[effective['weight_function']], effective, metadata,
                                 np.asarray([0]), np.asarray([1]), [nsyn['value']])[0, 0]
        stored = np.asarray(value).astype(actual.dtype)
        result['source_rows'].append(dict(global_row_id=edge['global_row_id'], edge_type_id=edge['edge_type_id'],
            nsyns=nsyn['value'], within_original_allowed_nsyns=nsyn['value'] in witness['allowed_nsyns'],
            table_row=row, effective_parameters=effective,
            source_population_predicate_matches=weights.parse_query(row['source_query'], 'pop_name') == witness['source_population'],
            target_type_predicate_matches=weights.parse_query(row['target_query'], 'node_type_id') == witness['target_node_type_id'],
            before_storage=numeric_record(value), after_storage=numeric_record(stored.item()),
            exact=bool(stored == actual), absolute_residual=float(abs(float(stored)-float(actual)))))
    return result


def associate(source_root: Path, arithmetic_report: Path, sonata_report: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_SOURCE_WEIGHT_ASSOCIATION', exit_code=1,
        scientific_qualification=False, production_episodes=0, full_graph_identity_qualified=False,
        scope='Three existing arithmetic witnesses with stored SONATA edge types and synapse counts; retained node metadata unchanged.',
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        arithmetic = json.loads(arithmetic_report.read_text())
        sonata = json.loads(sonata_report.read_text())
        if (arithmetic.get('status') != 'SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE'
                or sonata.get('status') != 'SOURCE_SONATA_WITNESS_QUERY_COMPLETE'
                or sonata.get('completed_selectors') != 3
                or sonata.get('witness_report_sha256') != sha256_file(arithmetic_report)
                or sonata.get('comparison_report_sha256') != arithmetic.get('comparison_report_sha256')):
            raise ValueError('ASSOCIATION_BOUND_COMPLETE_REPORTS_REQUIRED')
        sources = weights.load_sources(source_root)
        if sources['source_hashes'] != arithmetic['source_sha256']:
            raise ValueError('ASSOCIATION_SOURCE_HASHES_CHANGED')
        by_type = {row['edge_type_id']: row for row in sources['table'].values()}
        if len(by_type) != len(sources['table']):
            raise ValueError('ASSOCIATION_DUPLICATE_EDGE_TYPE')
        witnesses = {row['name']: row for row in arithmetic['cases']}
        if set(witnesses) != {row['name'] for row in sonata['cases']} or len(witnesses) != 3:
            raise ValueError('ASSOCIATION_WITNESS_SET_DIFFERS')
        cases = [associate_case(witnesses[case['name']], case, by_type, sources['rules']) for case in sonata['cases']]
        rows = [row for case in cases for row in case['source_rows']]
        report.update(status='SOURCE_WEIGHT_ASSOCIATION_DIAGNOSTIC_COMPLETE', exit_code=0, cases=cases,
            exact_source_rows=sum(row['exact'] for row in rows), source_rows=len(rows),
            target_type_predicate_conflicts=sum(not row['target_type_predicate_matches'] for row in rows),
            arithmetic_report_sha256=sha256_file(arithmetic_report), sonata_report_sha256=sha256_file(sonata_report),
            source_sha256=sources['source_hashes'], comparison_report_sha256=arithmetic['comparison_report_sha256'])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest({}, list(sys.argv), report['exit_code'])
        freeze_json(output/'report.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','graph','graph-inspection','node-validation','nodes','types',
                 'comparison-archive','config','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--arithmetic-report',type=Path)
    parser.add_argument('--sonata-report',type=Path)
    options=vars(parser.parse_args())
    if options['arithmetic_report'] is not None or options['sonata_report'] is not None:
        required=('source_root','arithmetic_report','sonata_report','output')
        if any(options[key] is None for key in required) or any(value is not None for key,value in options.items() if key not in required):
            parser.error('Association requires only source-root, arithmetic-report, sonata-report and output.')
        report=associate(**{key:options[key] for key in required})
    else:
        options.pop('arithmetic_report');options.pop('sonata_report')
        if any(value is None for value in options.values()):parser.error('All original arithmetic input paths are required.')
        report=diagnose(**options)
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))
    return report['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
