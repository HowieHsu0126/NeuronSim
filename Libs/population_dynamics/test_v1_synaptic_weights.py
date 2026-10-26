"""Small bound-source fixtures; no native execution or scientific circuit samples."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pandas as pd

from . import v1_synaptic_weights as weights
from .provenance import freeze_json, sha256_file

SOURCE_RULES = """
import nonexistent_nest_initialization
raise RuntimeError('module initialization must not execute')
@unavailable_registration
 def_invalid = 0
""".replace('@unavailable_registration\n def_invalid = 0', '') + """
@unavailable_registration
def DirectionRule_others(edges, src_nodes, trg_nodes):
    src_tuning = src_nodes['tuning_angle'].values
    tar_tuning = trg_nodes['tuning_angle'].values
    sigma = edges['weight_sigma'].values
    nsyn = edges['nsyns'].values
    syn_weight = edges['syn_weight'].values
    delta_tuning_180 = np.abs(np.abs(np.mod(np.abs(tar_tuning-src_tuning), 360.0)-180.0)-180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180/sigma)**2)
    return syn_weight*w_multiplier_180*nsyn

@unavailable_registration
def DirectionRule_EE(edges, src_nodes, trg_nodes):
    src_tuning = src_nodes['tuning_angle'].values
    tar_tuning = trg_nodes['tuning_angle'].values
    x_tar = trg_nodes['x'].values
    x_src = src_nodes['x'].values
    z_tar = trg_nodes['z'].values
    z_src = src_nodes['z'].values
    sigma = edges['weight_sigma'].values
    nsyn = edges['nsyns'].values
    syn_weight = edges['syn_weight'].values
    delta_tuning_180 = np.abs(np.abs(np.mod(np.abs(tar_tuning-src_tuning), 360.0)-180.0)-180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180/sigma)**2)
    delta_x = (x_tar-x_src)*0.07
    delta_z = (z_tar-z_src)*0.04
    theta_pref = tar_tuning*(np.pi/180.)
    xz = delta_x*np.cos(theta_pref)+delta_z*np.sin(theta_pref)
    sigma_phase = 1.0
    phase_scale_ratio = np.exp(-(xz**2/(2*sigma_phase**2)))
    theta_tar_scale = abs(abs(abs(180.0-np.mod(np.abs(tar_tuning), 360.0))-90.0)-90.0)
    phase_scale_ratio = phase_scale_ratio*(5.5/4.0-11.0/1680.0*theta_tar_scale)
    return syn_weight*w_multiplier_180*phase_scale_ratio*nsyn
"""
PAIR_RULE = """
raise RuntimeError('pair module initialization must not execute')
def compute_pair_type_parameters(source_type, target_type):
    unused_intercept = np.NaN
    return {'nsyn_range': cc_prob_dict[source_type+'-'+target_type]['nsyn_range']}
"""


def source(directory, value):
    directory.mkdir(parents=True)
    body = value.encode('utf-8') if isinstance(value, str) else json.dumps(value).encode('utf-8')
    (directory/'source_asset.txt').write_bytes(body)
    freeze_json(directory/'source.json', dict(artifact='source_asset.txt', bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(), synthetic_fixture=True))


def table_text(rows):
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=weights.TABLE_COLUMNS, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fixtures(root, mismatch=False, missing_pair=False):
    source_root = root/'source'
    populations = {11: 'e23Cux2', 22: 'e4Rorb', 33: 'i23Pvalb'}
    rows = []
    for target, target_pop in populations.items():
        for source_pop in populations.values():
            family = source_pop[0]+'2'+target_pop[0]
            rows.append(dict(edge_type_id=len(rows), target_query="node_type_id=='"+str(target)+"'",
                source_query="pop_name=='"+source_pop+"'", syn_weight=1.0 if source_pop[0] == 'e' else -0.125,
                weight_function='DirectionRule_EE' if family == 'e2e' else 'DirectionRule_others',
                weight_sigma=30.0, delay=1.8, dynamics_params=family+'.json', model_template='static_synapse'))
    source(source_root/'original_edge_functions', PAIR_RULE)
    source(source_root/'original_connection_parameters',
           {a+'-'+b: {'nsyn_range': [3, 5]} for a in populations.values() for b in populations.values()})
    source(source_root/'original_lgn_connection_parameters', {'synthetic_fixture': True})
    current_rows = deepcopy(rows)
    for row in current_rows[:2]:
        row['syn_weight'] += 0.25
    source(source_root/'original_glif_recurrent_parameters', table_text(current_rows))
    source(source_root/'weight_semantics/legacy_pointnet_runner', SOURCE_RULES)
    legacy_rows = [row for row in rows if not (missing_pair and row['source_query'] == "pop_name=='e23Cux2'"
                                              and row['target_query'] == "node_type_id=='22'")]
    source(source_root/'weight_semantics/legacy_recurrent_table', table_text(legacy_rows))
    source(source_root/'weight_semantics/legacy_glif_converter', '# Synthetic bound provenance only.\n')
    for name, receptor in (('e2e', 1), ('e2i', 3), ('i2e', 2), ('i2i', 4)):
        source(source_root/'weight_semantics'/('legacy_'+name), {'receptor_type': receptor})
    nodes, types = root/'nodes.h5', root/'types.txt'
    with h5py.File(nodes, 'w') as out:
        group = out.create_group('nodes/v1')
        for name, values in (('node_id', [0, 1, 2]), ('node_type_id', [11, 22, 33]),
                             ('node_group_id', [0, 0, 0]), ('node_group_index', [0, 1, 2])):
            group.create_dataset(name, data=np.asarray(values, dtype=np.int64))
        attributes = group.create_group('0')
        for name, values in (('x', [0., 5., 0.]), ('y', [0., 0., 0.]),
                             ('z', [0., 0., 0.]), ('tuning_angle', [0., 0., 0.])):
            attributes.create_dataset(name, data=np.asarray(values, dtype=np.float64))
    types.write_text('node_type_id pop_name\n'+''.join(str(key)+' '+value+'\n' for key, value in populations.items()))
    phase = np.exp(-((5.0*0.07)**2)/(2*1.0**2))*(5.5/4.0)
    recurrent = [dict(source=np.array([0, 1, 0]), target=np.array([1, 0, 1]),
        params=dict(receptor_type=1, weight=np.asarray([phase*3, phase*4, phase*3], dtype=np.float32),
                    delay=np.asarray([1.8, 1.8, 1.8], dtype=np.float32))),
        dict(source=np.array([2]), target=np.array([0]),
             params=dict(receptor_type=2, weight=np.array([-0.375], dtype=np.float64), delay=np.float64(1.8))),
        dict(source=np.array([0]), target=np.array([2]),
             params=dict(receptor_type=3, weight=np.array([4.0], dtype=np.float32), delay=np.float32(1.8)))]
    if mismatch:
        recurrent[0]['params']['weight'][2] += np.float32(0.25)
        recurrent[1]['params']['receptor_type'] = 4
        recurrent[1]['params']['delay'] = np.float64(2.0)
    graph = root/'graph.pkl'
    with graph.open('wb') as stream:
        pickle.dump({'nodes': [{'ids': np.array([i])} for i in range(3)], 'edges': recurrent}, stream)
    validation = root/'node_validation.json'
    freeze_json(validation, dict(status='NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS',
        nodes_sha256=sha256_file(nodes), node_types_sha256=sha256_file(types), cortical_units=3, source_cell_models=3))
    inspection = root/'graph_identity.json'
    freeze_json(inspection, dict(status='SOURCE_GRAPH_INSPECTED', native_metadata_identity_verified=True,
        cortical_neurons=3, raw_synapses=5, reference_metadata_sha256=sha256_file(validation),
        member_sha256={weights.GRAPH_MEMBER: sha256_file(graph), weights.NODES_MEMBER: sha256_file(nodes)},
        node_groups=[dict(raw_group_index=i, native_node_type_id=kind, neurons=1) for i, kind in enumerate(populations)]))
    return dict(source_root=source_root, graph=graph, graph_inspection=inspection,
                node_validation=validation, nodes=nodes, types=types, output=root/'output', batch_size=2)


class SynapticWeightTests(unittest.TestCase):
    def candidate(self, value, raw, candidates=(3, 4)):
        rules, _ = weights.extract_functions(SOURCE_RULES, weights.RULE_NAMES)
        nodes = pd.DataFrame({'tuning_angle': np.zeros(len(raw)), 'x': np.zeros(len(raw)), 'z': np.zeros(len(raw))})
        return weights.candidate_comparison(rules['DirectionRule_others'], nodes, nodes,
                                            {'syn_weight': value, 'weight_sigma': 30.0}, raw, candidates)

    def test_extraction_ignores_nest_initialization_and_registration(self):
        functions, adaptations = weights.extract_functions(SOURCE_RULES, weights.RULE_NAMES)
        self.assertEqual(set(functions), set(weights.RULE_NAMES))
        self.assertEqual(adaptations, 0)
        pair, adaptations = weights.extract_functions(PAIR_RULE, ('compute_pair_type_parameters',),
            alias_nan=True, namespace={'cc_prob_dict': {'e-e': {'nsyn_range': [3, 5]}}})
        self.assertEqual(adaptations, 1)
        self.assertEqual(weights.allowed_multiplicities(pair['compute_pair_type_parameters'], 'e', 'e'), (3, 4))
        self.assertIn('np.NaN', PAIR_RULE)

    def test_zero_underflow_ambiguity_and_signed_zero_are_distinct(self):
        raw = np.array([0.0, -0.0], dtype=np.float32)
        underflow = self.candidate(1e-46, raw)
        self.assertEqual(underflow['candidate_mask'].tolist(), [3, 3])
        self.assertEqual(underflow['zero_signbit_candidate_mask'].tolist(), [3, 0])
        self.assertEqual(underflow['pre_storage_zero_candidate_evaluations'], 0)
        self.assertEqual(underflow['storage_cast_to_zero_candidate_evaluations'], 4)
        zero = self.candidate(0.0, raw)
        self.assertEqual(zero['pre_storage_zero_candidate_evaluations'], 4)
        self.assertEqual(zero['storage_cast_to_zero_candidate_evaluations'], 0)
        smallest = np.nextafter(np.float32(0), np.float32(1))
        collision = self.candidate(5e-46, np.array([smallest], dtype=np.float32))
        self.assertEqual(collision['candidate_mask'].tolist(), [3])
        self.assertFalse(collision['raw_zero'][0])

    def test_actual_raw_dtype_controls_exact_candidate_comparison(self):
        rounded = np.array([0.1*3], dtype=np.float32)
        self.assertEqual(self.candidate(0.1, rounded)['candidate_mask'].tolist(), [1])
        result = self.candidate(0.1, rounded.astype(np.float64))
        self.assertEqual(result['candidate_mask'].tolist(), [0])
        self.assertGreater(result['nearest_absolute_residual'][0], 0)
        with self.assertRaisesRegex(ValueError, 'RAW_WEIGHT_FLOAT32_OR_FLOAT64_REQUIRED'):
            self.candidate(1.0, np.array([3], dtype=np.int64))

    def test_queries_duplicates_and_missing_rows_fail_closed(self):
        for query in ("pop_name=='e23Cux2' or True", "node_type_id=='1'", 'pop_name=e23Cux2'):
            with self.subTest(query=query), self.assertRaises(ValueError):
                weights.parse_query(query, 'pop_name')
        with tempfile.TemporaryDirectory(prefix='synthetic-weight-table-') as temporary:
            arguments = fixtures(Path(temporary), missing_pair=True)
            with self.assertRaisesRegex(ValueError, 'SOURCE_TABLE_PAIR_MISSING'):
                weights.compare(**arguments)
            failed = json.loads((arguments['output']/'report.json').read_text())
            self.assertEqual(failed['exit_code'], 1)
        row = dict(edge_type_id=1, source_query="pop_name=='e'", target_query="node_type_id=='1'",
            syn_weight=1.0, weight_function='DirectionRule_others', weight_sigma=30.0, delay=0.0,
            dynamics_params='e2e.json', model_template='static_synapse')
        # The acquired legacy table has 37 zero-delay declarations. Reading those
        # declarations must not substitute a positive delay or reject the table.
        self.assertEqual(weights.parse_table(table_text([row]))[('e', 1)]['delay'], 0.0)
        for altered in (dict(row, delay=-0.1), dict(row, weight_sigma=0.0)):
            with self.subTest(altered=altered), self.assertRaisesRegex(ValueError, 'SOURCE_TABLE_NUMBER_INVALID'):
                weights.parse_table(table_text([altered]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_OR_INVALID_IDENTITY'):
            weights.parse_table(table_text([row]*2))

    def test_complete_tuple_associations_and_literal_delay_rounding(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-weight-exact-') as temporary:
            arguments = fixtures(Path(temporary))
            report = weights.compare(**arguments)
            with (arguments['output']/'per_pair.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(report['status'], 'SOURCE_WEIGHT_RULE_CANDIDATES_EXACT_DEVELOPMENT')
        self.assertEqual(report['table_declarations']['changed_rows'], 2)
        self.assertEqual(report['table_declarations']['changed_fields'], {'syn_weight': 2})
        self.assertEqual(report['table_declarations']['selected_for_tuple_comparison'], 'legacy_recurrent_table')
        summary = report['comparison']
        self.assertEqual(summary['tuples'], 5)
        self.assertEqual(summary['unique_candidate_tuples'], 5)
        self.assertEqual(summary['delay_mismatches'], 0)
        self.assertEqual(summary['literal_delay_mismatches'], 4)
        self.assertGreater(summary['literal_delay_max_absolute_difference'], 0)
        duplicate_pair = next(row for row in rows if row['source_population'] == 'e23Cux2'
                              and row['target_node_type_id'] == '22')
        self.assertEqual(int(duplicate_pair['tuples']), 2)
        self.assertEqual(json.loads(duplicate_pair['candidate_mask_histogram']), {'1': 2})
        self.assertEqual(sum(int(row['tuples']) for row in rows), 5)
        self.assertFalse(report['original_nsyns_recovered'])
        self.assertFalse(report['source_fidelity_qualified'])
        self.assertEqual(report['production_episodes'], 0)

    def test_numerical_mismatches_preserve_complete_denominator(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-weight-differs-') as temporary:
            arguments = fixtures(Path(temporary), mismatch=True)
            report = weights.compare(**arguments)
        self.assertEqual(report['exit_code'], 2)
        self.assertEqual(report['comparison']['tuples'], 5)
        self.assertEqual(report['comparison']['no_candidate_tuples'], 1)
        self.assertEqual(report['comparison']['receptor_mismatches'], 1)
        self.assertEqual(report['comparison']['delay_mismatches'], 1)
        locations = {(row['raw_group'], row['raw_offset']) for row in report['comparison']['mismatch_examples']}
        self.assertEqual(locations, {(0, 2), (1, 0)})

    def test_graph_and_node_hashes_are_checked_before_unpickle(self):
        for altered in ('graph', 'types'):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory(prefix='synthetic-weight-binding-') as temporary:
                arguments = fixtures(Path(temporary))
                with arguments[altered].open('ab') as stream:
                    stream.write(b'changed')
                with mock.patch.object(weights.pickle, 'load') as unpickle:
                    with self.assertRaises(ValueError):
                        weights.compare(**arguments)
                    unpickle.assert_not_called()
                report = json.loads((arguments['output']/'report.json').read_text())
                self.assertEqual(report['exit_code'], 1)
                self.assertFalse(report['scientific_qualification'])


if __name__ == '__main__':
    unittest.main()
