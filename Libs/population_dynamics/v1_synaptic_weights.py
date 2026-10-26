"""Compare source GLIF weight candidates with every hash-bound raw graph tuple.

This tests compatibility at the actual stored precision. It neither recovers
historical SONATA nsyns nor qualifies a scientific circuit family.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_circuit_source import read_source
from .v1_graph_identity import _h5_metadata, _types
from .v1_nest_network import edge_batches

RULE_NAMES = ('DirectionRule_others', 'DirectionRule_EE')
TABLE_COLUMNS = ('edge_type_id', 'target_query', 'source_query', 'syn_weight',
                 'weight_function', 'weight_sigma', 'delay', 'dynamics_params', 'model_template')
SOURCE_NAMES = ('original_edge_functions', 'original_connection_parameters',
    'original_lgn_connection_parameters', 'original_glif_recurrent_parameters',
    'weight_semantics/legacy_pointnet_runner', 'weight_semantics/legacy_recurrent_table',
    'weight_semantics/legacy_glif_converter', 'weight_semantics/legacy_e2e',
    'weight_semantics/legacy_e2i', 'weight_semantics/legacy_i2e', 'weight_semantics/legacy_i2i')
GRAPH_MEMBER = 'GLIF_network2/network_dat.pkl'
NODES_MEMBER = 'GLIF_network2/network/v1_nodes.h5'
MAX_MISMATCH_EXAMPLES = 16


class _NanAlias(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'NaN':
            node.attr = 'nan'
            self.count += 1
        return node


def extract_functions(source: str, names: tuple[str, ...], *, alias_nan=False, namespace=None):
    """Compile copied function definitions only; do not execute module initialization."""
    tree = ast.parse(source)
    functions = []
    adaptation = _NanAlias()
    for name in names:
        found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(found) != 1:
            raise ValueError('SOURCE_FUNCTION_MISSING_OR_DUPLICATE:' + name)
        node = copy.deepcopy(found[0])
        if (node.args.defaults or any(value is not None for value in node.args.kw_defaults)
                or node.returns is not None
                or any(arg.annotation is not None for arg in (*node.args.posonlyargs, *node.args.args,
                                                              *node.args.kwonlyargs))):
            raise ValueError('SOURCE_FUNCTION_DEFINITION_EVALUATION_UNEXPECTED:' + name)
        node.decorator_list = []
        functions.append(adaptation.visit(node) if alias_nan else node)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    scope = {'np': np}
    if namespace:
        scope.update(namespace)
    exec(compile(module, '<bound-source-functions>', 'exec'), scope)
    return {name: scope[name] for name in names}, adaptation.count


def parse_query(value, field):
    if not isinstance(value, str):
        raise ValueError('SOURCE_TABLE_QUERY_REQUIRED:' + field)
    match = re.fullmatch(r"([a-z_]+)==(['\"])([^'\"]+)\2", value)
    if match is None or match[1] != field:
        raise ValueError('SOURCE_TABLE_QUERY_INVALID:' + value)
    result = match[3]
    if field == 'node_type_id':
        if not re.fullmatch(r'[0-9]+', result):
            raise ValueError('SOURCE_TABLE_NODE_TYPE_INVALID:' + result)
        return int(result)
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', result):
        raise ValueError('SOURCE_TABLE_POPULATION_INVALID:' + result)
    return result


def parse_table(source: str):
    # Match the author's converter parser, including its default float parser.
    frame = pd.read_csv(io.StringIO(source), sep=' ')
    if set(frame.columns) != set(TABLE_COLUMNS) or frame.empty:
        raise ValueError('SOURCE_TABLE_COLUMNS_OR_ROWS_INVALID')
    result, edge_ids = {}, set()
    for original in frame.to_dict(orient='records'):
        key = (parse_query(original['source_query'], 'pop_name'),
               parse_query(original['target_query'], 'node_type_id'))
        edge_id = original['edge_type_id']
        if (isinstance(edge_id, bool) or not isinstance(edge_id, (int, np.integer))
                or edge_id < 0 or edge_id in edge_ids or key in result):
            raise ValueError('SOURCE_TABLE_DUPLICATE_OR_INVALID_IDENTITY:' + repr(key))
        row = dict(original, edge_type_id=int(edge_id))
        for field in ('syn_weight', 'weight_sigma', 'delay'):
            try:
                number = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError('SOURCE_TABLE_NUMBER_INVALID:' + field) from error
            # The legacy source declares zero delay for some type pairs. Preserve
            # it for comparison; native edge admission still requires positive delay.
            if (not np.isfinite(number) or (field == 'weight_sigma' and number <= 0)
                    or (field == 'delay' and number < 0)):
                raise ValueError('SOURCE_TABLE_NUMBER_INVALID:' + field)
            row[field] = number
        if row['weight_function'] not in RULE_NAMES or row['model_template'] != 'static_synapse':
            raise ValueError('SOURCE_TABLE_WEIGHT_RULE_OR_SYNAPSE_INVALID:' + repr(key))
        if row['dynamics_params'] not in ('e2e.json', 'e2i.json', 'i2e.json', 'i2i.json'):
            raise ValueError('SOURCE_TABLE_RECEPTOR_SOURCE_INVALID:' + repr(key))
        result[key] = row
        edge_ids.add(int(edge_id))
    return result


def table_differences(current, legacy):
    common = set(current) & set(legacy)
    fields, examples = Counter(), []
    changed_rows = 0
    for key in sorted(common):
        differences = {name: {'current': current[key][name], 'legacy': legacy[key][name]}
                       for name in TABLE_COLUMNS if current[key][name] != legacy[key][name]}
        if differences:
            changed_rows += 1
            fields.update(differences.keys())
            if len(examples) < MAX_MISMATCH_EXAMPLES:
                examples.append(dict(source_population=key[0], target_node_type_id=key[1], fields=differences))
    return dict(current_rows=len(current), legacy_rows=len(legacy), common_rows=len(common),
        changed_rows=changed_rows, changed_fields=dict(fields),
        current_only=[list(key) for key in sorted(set(current)-set(legacy))],
        legacy_only=[list(key) for key in sorted(set(legacy)-set(current))], examples=examples,
        selected_for_tuple_comparison='legacy_recurrent_table', outcome_based_table_selection=False)


def load_sources(source_root: Path):
    raw, identities = {}, {}
    for name in SOURCE_NAMES:
        body, receipt = read_source(source_root/name)
        raw[name] = body.decode('utf-8')
        identities[name] = dict(sha256=receipt['sha256'], bytes=receipt['bytes'],
                               receipt_sha256=sha256_file(source_root/name/'source.json'))
    rules, _ = extract_functions(raw['weight_semantics/legacy_pointnet_runner'], RULE_NAMES)
    compute, adaptations = extract_functions(raw['original_edge_functions'],
        ('compute_pair_type_parameters',), alias_nan=True,
        namespace={'cc_prob_dict': json.loads(raw['original_connection_parameters']),
                   'logger': logging.getLogger(__name__)})
    receptors = {}
    for name in ('e2e', 'e2i', 'i2e', 'i2i'):
        value = json.loads(raw['weight_semantics/legacy_'+name]).get('receptor_type')
        if type(value) is not int or not 1 <= value <= 4:
            raise ValueError('SOURCE_RECEPTOR_DECLARATION_INVALID:' + name)
        receptors[name+'.json'] = value
    legacy = parse_table(raw['weight_semantics/legacy_recurrent_table'])
    current = parse_table(raw['original_glif_recurrent_parameters'])
    return dict(rules=rules, compute=compute['compute_pair_type_parameters'], receptors=receptors,
                table=legacy, source_hashes=identities, nan_alias_adaptations=adaptations,
                table_comparison=table_differences(current, legacy))


def allowed_multiplicities(compute, source_population, target_population):
    values = compute(source_population, target_population)['nsyn_range']
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(type(value) is not int for value in values)
            or not 1 <= values[0] < values[1]):
        raise ValueError('SOURCE_NSYNS_INTEGER_RANGE_INVALID')
    return tuple(range(values[0], values[1]))


def candidate_comparison(rule, source_nodes, target_nodes, table_row, weights, candidates):
    weights = np.asarray(weights)
    if (weights.ndim != 1 or weights.dtype.kind != 'f' or weights.dtype.itemsize not in (4, 8)
            or not np.isfinite(weights).all() or not candidates):
        raise ValueError('RAW_WEIGHT_FLOAT32_OR_FLOAT64_REQUIRED')
    length = len(weights)
    mask_type = np.uint64 if len(candidates) <= 64 else int
    masks = np.zeros(length, dtype=np.uint64 if len(candidates) <= 64 else object)
    sign_masks = masks.copy()
    nearest = np.full(length, np.inf, dtype=np.float64)
    source_zero, storage_zero = 0, 0
    edges = pd.DataFrame({'syn_weight': np.full(length, table_row['syn_weight']),
                          'weight_sigma': np.full(length, table_row['weight_sigma']),
                          'nsyns': np.zeros(length, dtype=np.int64)})
    for bit, nsyns in enumerate(candidates):
        edges['nsyns'] = nsyns
        before_cast = np.asarray(rule(edges, source_nodes, target_nodes))
        if before_cast.shape != weights.shape or not np.isfinite(before_cast).all():
            raise ValueError('SOURCE_WEIGHT_FUNCTION_INVALID_OUTPUT')
        with np.errstate(under='ignore', over='ignore', invalid='ignore'):
            predicted = before_cast.astype(weights.dtype)
        if not np.isfinite(predicted).all():
            raise ValueError('SOURCE_WEIGHT_STORAGE_CAST_NONFINITE')
        equal = predicted == weights
        masks[equal] |= mask_type(1 << bit)
        sign_equal = equal & (weights == 0) & (np.signbit(predicted) == np.signbit(weights))
        sign_masks[sign_equal] |= mask_type(1 << bit)
        source_zero += int(np.count_nonzero(before_cast == 0))
        storage_zero += int(np.count_nonzero((before_cast != 0) & (predicted == 0)))
        with np.errstate(over='ignore'):
            residual = np.abs(weights.astype(np.float64)-predicted.astype(np.float64))
        nearest = np.minimum(nearest, residual)
    if not np.isfinite(nearest).all():
        raise ValueError('NEAREST_CANDIDATE_RESIDUAL_NOT_FINITE')
    return dict(candidate_mask=masks, zero_signbit_candidate_mask=sign_masks,
        nearest_absolute_residual=nearest, raw_zero=weights == 0,
        pre_storage_zero_candidate_evaluations=source_zero,
        storage_cast_to_zero_candidate_evaluations=storage_zero,
        raw_weight_dtype=weights.dtype.str)


def _histogram(values):
    unique, counts = np.unique(values, return_counts=True)
    return Counter({int(value): int(count) for value, count in zip(unique, counts)})


def _new_pair(key, row, target_population, candidates, receptor):
    return dict(source_population=key[0], target_node_type_id=key[1],
        target_population=target_population, source_edge_type_id=row['edge_type_id'],
        weight_function=row['weight_function'], source_syn_weight=row['syn_weight'],
        source_weight_sigma=row['weight_sigma'], source_delay=row['delay'],
        source_receptor=receptor, allowed_nsyns=list(candidates), tuples=0,
        unique_candidate_tuples=0, ambiguous_candidate_tuples=0, no_candidate_tuples=0,
        zero_weight_tuples=0, zero_signbit_unmatched_tuples=0,
        delay_mismatches=0, receptor_mismatches=0, literal_delay_mismatches=0,
        literal_delay_max_absolute_difference=0.0,
        nearest_candidate_absolute_residual_sum=0.0, nearest_candidate_absolute_residual_max=0.0,
        pre_storage_zero_candidate_evaluations=0, storage_cast_to_zero_candidate_evaluations=0,
        candidate_mask_histogram=Counter(), zero_candidate_mask_histogram=Counter(),
        zero_signbit_mask_histogram=Counter(), raw_weight_dtypes=set(), raw_delay_dtypes=set())


def compare_tuples(raw, metadata, types, graph_report, sources, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('POSITIVE_BATCH_SIZE_REQUIRED')
    n = graph_report['cortical_neurons']
    if len(metadata['node_id']) != n:
        raise ValueError('GRAPH_AND_NODE_DENOMINATORS_DIFFER')
    for name in ('x', 'z', 'tuning_angle'):
        if metadata[name].shape != (n,) or not np.isfinite(metadata[name]).all():
            raise ValueError('SOURCE_NODE_WEIGHT_ATTRIBUTE_INVALID:' + name)
    seen = np.zeros(n, dtype=bool)
    node_groups = graph_report['node_groups']
    if len(raw['nodes']) != len(node_groups):
        raise ValueError('RAW_NODE_GROUP_DENOMINATOR_DIFFERS')
    for index, (group, identity) in enumerate(zip(raw['nodes'], node_groups)):
        ids = np.asarray(group['ids'])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= n).any()
                or len(np.unique(ids)) != len(ids) or seen[ids].any()
                or identity['raw_group_index'] != index or len(ids) != identity['neurons']
                or not np.all(metadata['node_type_id'][ids] == identity['native_node_type_id'])):
            raise ValueError('RAW_NODE_GROUP_IDENTITY_DIFFERS')
        seen[ids] = True
    if not seen.all() or set(map(int, metadata['node_type_id'])) != set(types):
        raise ValueError('SOURCE_NODE_TYPE_COVERAGE_DIFFERS')
    type_names = sorted(types)
    populations = sorted({row['pop_name'] for row in types.values()})
    population_codes = {name: index for index, name in enumerate(populations)}
    type_codes = {name: index for index, name in enumerate(type_names)}
    source_codes = np.asarray([population_codes[types[int(value)]['pop_name']]
                               for value in metadata['node_type_id']], dtype=np.int64)
    target_codes = np.asarray([type_codes[int(value)] for value in metadata['node_type_id']], dtype=np.int64)
    positive_ids = np.arange(1, n+1, dtype=np.int64)
    table = sources['table']
    pairs, range_cache = {}, {}
    for key, row in sorted(table.items()):
        if key[0] not in population_codes or key[1] not in types:
            raise ValueError('SOURCE_TABLE_UNKNOWN_NODE_POPULATION:' + repr(key))
        target_population = types[key[1]]['pop_name']
        range_key = (key[0], target_population)
        if range_key not in range_cache:
            range_cache[range_key] = allowed_multiplicities(sources['compute'], *range_key)
        pairs[key] = _new_pair(key, row, target_population, range_cache[range_key],
                              sources['receptors'][row['dynamics_params']])
    compared, examples = 0, []
    for group_index, group in enumerate(raw['edges']):
        weights = np.asarray(group['params']['weight'])
        delays = np.asarray(group['params']['delay'])
        if (weights.dtype.kind != 'f' or weights.dtype.itemsize not in (4, 8)
                or delays.dtype.kind != 'f' or delays.dtype.itemsize not in (4, 8)):
            raise ValueError('RAW_WEIGHT_AND_DELAY_FLOAT32_OR_FLOAT64_REQUIRED')
        offset = 0
        for batch in edge_batches((group,), positive_ids, positive_ids, batch_size):
            source_ids, target_ids = batch['source']-1, batch['target']-1
            length = len(source_ids)
            raw_weights = weights[offset:offset+length]
            raw_delays = (np.full(length, delays.item(), dtype=delays.dtype) if delays.ndim == 0
                          else delays[offset:offset+length])
            keys = source_codes[source_ids]*len(type_names)+target_codes[target_ids]
            order = np.argsort(keys, kind='stable')
            boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order]))+1, length]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                selected = order[start:stop]
                code = int(keys[selected[0]])
                key = (populations[code//len(type_names)], type_names[code % len(type_names)])
                if key not in table:
                    raise ValueError('SOURCE_TABLE_PAIR_MISSING:' + repr(key))
                row, pair = table[key], pairs[key]
                src, dst = source_ids[selected], target_ids[selected]
                source_frame = pd.DataFrame({name: metadata[name][src] for name in ('x', 'z', 'tuning_angle')})
                target_frame = pd.DataFrame({name: metadata[name][dst] for name in ('x', 'z', 'tuning_angle')})
                actual_weights, actual_delays = raw_weights[selected], raw_delays[selected]
                comparison = candidate_comparison(sources['rules'][row['weight_function']],
                    source_frame, target_frame, row, actual_weights, pair['allowed_nsyns'])
                masks = comparison['candidate_mask']
                histogram = _histogram(masks)
                zero = comparison['raw_zero']
                sign_masks = comparison['zero_signbit_candidate_mask']
                pair['candidate_mask_histogram'].update(histogram)
                pair['zero_candidate_mask_histogram'].update(_histogram(masks[zero]))
                pair['zero_signbit_mask_histogram'].update(_histogram(sign_masks[zero]))
                pair['tuples'] += len(selected)
                pair['no_candidate_tuples'] += histogram.get(0, 0)
                pair['unique_candidate_tuples'] += sum(count for mask, count in histogram.items() if mask.bit_count() == 1)
                pair['ambiguous_candidate_tuples'] += sum(count for mask, count in histogram.items() if mask.bit_count() > 1)
                pair['zero_weight_tuples'] += int(zero.sum())
                pair['zero_signbit_unmatched_tuples'] += int(np.count_nonzero(zero & (masks != 0) & (sign_masks == 0)))
                expected_delay = np.asarray(row['delay'], dtype=actual_delays.dtype)
                delay_equal = actual_delays == expected_delay
                receptor_equal = batch['receptor_type'] == pair['source_receptor']
                literal_difference = np.abs(actual_delays.astype(np.float64)-row['delay'])
                pair['delay_mismatches'] += int(np.count_nonzero(~delay_equal))
                pair['receptor_mismatches'] += 0 if receptor_equal else len(selected)
                pair['literal_delay_mismatches'] += int(np.count_nonzero(literal_difference))
                pair['literal_delay_max_absolute_difference'] = max(pair['literal_delay_max_absolute_difference'], float(literal_difference.max()))
                residual = comparison['nearest_absolute_residual']
                pair['nearest_candidate_absolute_residual_sum'] += float(residual.sum(dtype=np.float64))
                pair['nearest_candidate_absolute_residual_max'] = max(pair['nearest_candidate_absolute_residual_max'], float(residual.max()))
                for name in ('pre_storage_zero_candidate_evaluations', 'storage_cast_to_zero_candidate_evaluations'):
                    pair[name] += comparison[name]
                pair['raw_weight_dtypes'].add(actual_weights.dtype.str)
                pair['raw_delay_dtypes'].add(actual_delays.dtype.str)
                bad = (masks == 0) | ~delay_equal | (not receptor_equal)
                remaining = MAX_MISMATCH_EXAMPLES-len(examples)
                for position in np.flatnonzero(bad)[:remaining]:
                    examples.append(dict(raw_group=group_index, raw_offset=int(offset+selected[position]),
                        source_id=int(src[position]), target_id=int(dst[position]),
                        source_population=key[0], target_node_type_id=key[1],
                        raw_weight=float(actual_weights[position]), raw_weight_dtype=actual_weights.dtype.str,
                        raw_delay=float(actual_delays[position]), raw_delay_dtype=actual_delays.dtype.str,
                        source_delay=row['delay'], source_delay_after_storage_cast=float(expected_delay),
                        raw_receptor=batch['receptor_type'], source_receptor=pair['source_receptor'],
                        candidate_mask=int(masks[position]), allowed_nsyns=pair['allowed_nsyns'],
                        nearest_candidate_absolute_residual=float(residual[position])))
            offset += length
            compared += length
        if offset != len(weights):
            raise ValueError('RAW_GROUP_TUPLE_DENOMINATOR_DIFFERS')
    if compared != graph_report['raw_synapses']:
        raise ValueError('RAW_GRAPH_TUPLE_DENOMINATOR_DIFFERS')
    count_fields = ('tuples', 'unique_candidate_tuples', 'ambiguous_candidate_tuples',
        'no_candidate_tuples', 'zero_weight_tuples', 'zero_signbit_unmatched_tuples',
        'delay_mismatches', 'receptor_mismatches', 'literal_delay_mismatches',
        'pre_storage_zero_candidate_evaluations', 'storage_cast_to_zero_candidate_evaluations')
    summary = {name: sum(pair[name] for pair in pairs.values()) for name in count_fields}
    summary.update(declared_type_pairs=len(pairs), compared_type_pairs=sum(pair['tuples'] > 0 for pair in pairs.values()),
        literal_delay_max_absolute_difference=max(pair['literal_delay_max_absolute_difference'] for pair in pairs.values()),
        nearest_candidate_absolute_residual_max=max(pair['nearest_candidate_absolute_residual_max'] for pair in pairs.values()),
        nearest_candidate_absolute_residual_mean=sum(pair['nearest_candidate_absolute_residual_sum'] for pair in pairs.values())/compared,
        mismatch_examples=examples)
    for pair in pairs.values():
        pair['nearest_candidate_absolute_residual_mean'] = (pair['nearest_candidate_absolute_residual_sum']/pair['tuples']
                                                          if pair['tuples'] else None)
        for name in ('candidate_mask_histogram', 'zero_candidate_mask_histogram', 'zero_signbit_mask_histogram'):
            pair[name] = {str(mask): count for mask, count in sorted(pair[name].items())}
        for name in ('raw_weight_dtypes', 'raw_delay_dtypes'):
            pair[name] = sorted(pair[name])
    return summary, list(pairs.values())


def compare(source_root: Path, graph: Path, graph_inspection: Path, node_validation: Path,
            nodes: Path, types: Path, output: Path, batch_size=100000):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_SOURCE_WEIGHT_RULE_COMPARISON', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), source_fidelity_qualified=False,
        scientific_qualification=False, production_episodes=0, new_full_circuits=0,
        original_nsyns_recovered=False, numerical_tolerance=0,
        qualification_scope='Compatibility with legacy source weight candidates at actual stored precision only.',
        comparison_policy=dict(weight='Exact numerical equality after actual raw float32/float64 storage cast.',
            zero_signbit='Reported separately; numerical zero equality does not establish bitwise equality.',
            delay='Exact after table float64 to actual raw floating dtype cast; literal float64 difference also reported.',
            receptor='Exact acquired dynamics_params receptor_type; no inferred E/I mapping.',
            multiplicity='Complete source integer candidate set, upper-exclusive; no historical nsyns inference.',
            source_selection='Legacy table and exact legacy runner bodies, selected before outcomes; current table descriptive only.',
            tuple_identity='Original raw group/offset and source/target/receptor/delay association; no deduplication.'),
        batch_size=batch_size)
    try:
        inspection = json.loads(graph_inspection.read_text())
        if (inspection.get('status') != 'SOURCE_GRAPH_INSPECTED'
                or inspection.get('native_metadata_identity_verified') is not True
                or type(inspection.get('cortical_neurons')) is not int or inspection['cortical_neurons'] < 1
                or type(inspection.get('raw_synapses')) is not int or inspection['raw_synapses'] < 1):
            raise ValueError('VERIFIED_RAW_GRAPH_INSPECTION_REQUIRED')
        hashes = {name: sha256_file(path) for name, path in
                  (('graph', graph), ('graph_inspection', graph_inspection), ('node_validation', node_validation),
                   ('nodes', nodes), ('types', types))}
        if hashes['graph'] != inspection['member_sha256'][GRAPH_MEMBER]:
            raise ValueError('RAW_GRAPH_HASH_MISMATCH_BEFORE_UNPICKLE')
        if hashes['node_validation'] != inspection['reference_metadata_sha256']:
            raise ValueError('NODE_VALIDATION_HASH_MISMATCH')
        validation = json.loads(node_validation.read_text())
        if (validation.get('status') != 'NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS'
                or hashes['nodes'] != validation['nodes_sha256']
                or hashes['nodes'] != inspection['member_sha256'][NODES_MEMBER]
                or hashes['types'] != validation['node_types_sha256']
                or validation['cortical_units'] != inspection['cortical_neurons']):
            raise ValueError('SOURCE_NODE_METADATA_BINDING_DIFFERS')
        report['input_sha256'] = hashes
        sources = load_sources(source_root)
        report.update(source_sha256=sources['source_hashes'],
            source_function_adaptations={'weight_bodies': 'None; registration decorators removed only.',
                                        'pair_parameter_np_NaN_to_nan': sources['nan_alias_adaptations']},
            table_declarations=sources['table_comparison'], source_receptors=sources['receptors'])
        metadata, type_rows = _h5_metadata(nodes), _types(types)
        if len(type_rows) != validation['source_cell_models']:
            raise ValueError('SOURCE_CELL_TYPE_DENOMINATOR_DIFFERS')
        # The existing inspection is the trust anchor; never unpickle before its hash checks.
        with graph.open('rb') as stream:
            raw = pickle.load(stream)
        summary, rows = compare_tuples(raw, metadata, type_rows, inspection, sources, batch_size)
        destination = output/'per_pair.csv'
        with destination.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
        exact = not any(summary[key] for key in ('no_candidate_tuples', 'delay_mismatches', 'receptor_mismatches'))
        report.update(status='SOURCE_WEIGHT_RULE_CANDIDATES_EXACT_DEVELOPMENT' if exact else 'SOURCE_WEIGHT_RULE_DIFFERS_DIAGNOSTIC',
            exit_code=0 if exact else 2, comparison=summary,
            per_pair=dict(path='per_pair.csv', sha256=sha256_file(destination), rows=len(rows),
                          mask_bit_order='Ascending allowed_nsyns; decimal integer masks; zero means no candidate.'))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest({'batch_size': batch_size}, list(sys.argv), report['exit_code'])
        freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'graph', 'graph-inspection', 'node-validation', 'nodes', 'types', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=100000)
    args = parser.parse_args()
    return compare(**vars(args))['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
