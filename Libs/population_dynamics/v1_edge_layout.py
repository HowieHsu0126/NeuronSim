"""Inspect explicit raw-edge metadata and population grouping without fitting weights."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_graph_identity import _h5_metadata, _integers, _types


def describe_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return dict(type=type(value).__name__, value=value)
    if isinstance(value, float):
        return dict(type='float', value=value if np.isfinite(value) else None, repr=repr(value))
    if isinstance(value, dict):
        return dict(type='dict', keys=sorted(value))
    array = np.asarray(value)
    result = dict(type=type(value).__name__, shape=list(array.shape), dtype=str(array.dtype))
    if array.dtype.kind in 'iubUS' and array.size <= 16:
        result['values'] = array.tolist()
    return result


def describe_groups(raw, metadata, types):
    native_types = metadata['node_type_id']
    n = len(native_types)
    groups, extra_keys, parameter_keys = [], Counter(), Counter()
    for index, group in enumerate(raw['edges']):
        source = _integers(group['source'], 'edge_source')
        target = _integers(group['target'], 'edge_target')
        if (len(source) != len(target) or np.any(source < 0) or np.any(target < 0)
                or np.any(source >= n) or np.any(target >= n)):
            raise ValueError('V1_LAYOUT_EDGE_ENDPOINT_INVALID')
        extras = {key: describe_value(value) for key, value in group.items()
                  if key not in ('source', 'target', 'params')}
        extra_keys.update(extras.keys());parameter_keys.update(group['params'].keys())
        entry = dict(raw_group_index=index, tuples=len(source), group_keys=sorted(group),
            extra_fields=extras, parameter_fields={key: describe_value(value) for key,value in group['params'].items()})
        for role, ids in (('source',source),('target',target)):
            identities, counts = np.unique(native_types[ids], return_counts=True)
            populations = Counter()
            for identity,count in zip(identities,counts):
                populations[types[int(identity)]['pop_name']] += int(count)
            entry[role+'_node_types'] = [dict(node_type_id=int(identity),tuple_occurrences=int(count))
                                        for identity,count in zip(identities,counts)]
            entry[role+'_populations'] = dict(sorted(populations.items()))
        groups.append(entry)
    return dict(raw_root_keys=sorted(raw), cortical_units=n, raw_edge_groups=len(groups),
        raw_tuples=sum(row['tuples'] for row in groups), empty_groups=sum(row['tuples']==0 for row in groups),
        extra_group_key_counts=dict(extra_keys), parameter_key_counts=dict(parameter_keys),
        groups_with_multiple_source_populations=sum(len(row['source_populations'])>1 for row in groups),
        groups_with_multiple_target_populations=sum(len(row['target_populations'])>1 for row in groups),
        groups_with_multiple_target_node_types=sum(len(row['target_node_types'])>1 for row in groups),
        group_order_to_sonata_type_qualified=False, groups=groups)


def inspect_layout(graph: Path, nodes: Path, types: Path, comparison_archive: Path, output: Path):
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED_SOURCE_EDGE_LAYOUT_INSPECTION',exit_code=1,
        scientific_qualification=False,production_episodes=0,
        scope='All raw groups and explicit metadata; no weight-based type assignment, deduplication or simulation.',
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        reference=comparison_archive/'weights/report.json'
        delivery=json.loads((comparison_archive/'delivery.json').read_text())
        record=next(row for row in delivery['files'] if row['path']=='weights/report.json')
        if reference.stat().st_size!=record['bytes'] or sha256_file(reference)!=record['sha256']:
            raise ValueError('V1_LAYOUT_COMPARISON_DELIVERY_CHANGED')
        baseline=json.loads(reference.read_text())
        if baseline['status'] not in ('SOURCE_WEIGHT_RULE_DIFFERS_DIAGNOSTIC','SOURCE_WEIGHT_RULE_CANDIDATES_EXACT_DEVELOPMENT'):
            raise ValueError('V1_LAYOUT_COMPLETE_COMPARISON_REQUIRED')
        hashes={name:sha256_file(path) for name,path in [('graph',graph),('nodes',nodes),('types',types)]}
        if any(value!=baseline['input_sha256'][name] for name,value in hashes.items()):
            raise ValueError('V1_LAYOUT_INPUT_CHANGED_BEFORE_UNPICKLE')
        metadata, type_rows = _h5_metadata(nodes), _types(types)
        with graph.open('rb') as stream:
            raw=pickle.load(stream)
        layout=describe_groups(raw,metadata,type_rows)
        if layout['raw_tuples']!=baseline['comparison']['tuples']:
            raise ValueError('V1_LAYOUT_ORIGINAL_TUPLE_DENOMINATOR_CHANGED')
        groups=layout.pop('groups')
        freeze_json(output/'groups.json',groups)
        report.update(status='SOURCE_EDGE_LAYOUT_INSPECTED',exit_code=0,layout=layout,
            groups=dict(path='groups.json',sha256=sha256_file(output/'groups.json')),
            input_sha256=hashes,comparison_report_sha256=record['sha256'])
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        report['runtime']=runtime_manifest({},list(sys.argv),report['exit_code'])
        freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','nodes','types','comparison-archive','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    report=inspect_layout(**vars(parser.parse_args()))
    print(json.dumps(dict(status=report['status'],layout=report['layout'])))
    return report['exit_code']

if __name__=='__main__':
    raise SystemExit(main())
