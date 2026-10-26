"""Source arithmetic witness selection and lineage using small bound inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml
import numpy as np

from . import v1_synaptic_weights as weights
from . import v1_weight_arithmetic as arithmetic
from .provenance import freeze_json, sha256_file
from .test_v1_synaptic_weights import fixtures, source

BIO_RULES = """
raise RuntimeError('module initialization must not execute')
@unavailable_registration
def DirectionRule_others(edge_props,src_node,trg_node):
    return edge_props['syn_weight']
@unavailable_registration
def DirectionRule_EE(edge_props,src_node,trg_node):
    delta=(trg_node['x']-src_node['x'])*0.07
    return np.exp(-(delta**2)/2)*(5.5/4.0)*edge_props['syn_weight']
"""


def setup(root):
    arguments=fixtures(root,mismatch=True)
    original=weights.compare(**arguments)
    archive=root/'comparison';(archive/'weights').mkdir(parents=True)
    baseline=archive/'weights/report.json';baseline.write_bytes((arguments['output']/'report.json').read_bytes())
    freeze_json(archive/'delivery.json',{'files':[dict(path='weights/report.json',bytes=baseline.stat().st_size,
        sha256=sha256_file(baseline))]})
    source(arguments['source_root']/'weight_semantics/original_bionet_runner',BIO_RULES)
    config=root/'diagnostic.yaml'
    config.write_text(yaml.safe_dump({'cases':[
        dict(name='exact_location',selection='raw_location',raw_group=0,raw_offset=2,expected_source_id=0,expected_target_id=1),
        dict(name='first_pair',selection='first',source_population='e23Cux2',target_node_type_id=22),
        dict(name='max_pair',selection='max_absolute_residual',source_population='e23Cux2',target_node_type_id=22)]}))
    arguments.pop('batch_size');arguments.update(output=root/'diagnostic',comparison_archive=archive,config=config)
    return arguments,original,baseline


class WeightArithmeticTests(unittest.TestCase):
    def test_original_batch_first_and_max_witnesses_preserve_candidates(self):
        with tempfile.TemporaryDirectory(prefix='weight-arithmetic-') as temporary:
            arguments,original,baseline=setup(Path(temporary));before=baseline.read_bytes()
            report=arithmetic.diagnose(**arguments)
            self.assertEqual(before,baseline.read_bytes())
        self.assertEqual(report['status'],'SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE')
        self.assertEqual(original['status'],'SOURCE_WEIGHT_RULE_DIFFERS_DIAGNOSTIC')
        location,first,largest=report['cases']
        self.assertEqual([r['raw_offset'] for r in report['cases']],[2,0,2])
        self.assertEqual(location['original_batch_start'],2)
        self.assertEqual(first['original_pair_batch_tuples'],1)
        self.assertEqual(first['main_path_nearest_absolute_residual'],0)
        self.assertGreater(largest['main_path_nearest_absolute_residual'],0)
        for row in report['cases']:
            self.assertEqual(len(row['variants']),6)
            self.assertEqual(row['source_node']['node_group_index'],row['source_id'])
            for variant in row['variants']:
                self.assertEqual([c['nsyns'] for c in variant['candidates']],[3,4])
                self.assertIn('hex',variant['parsed_numbers']['syn_weight'])
        self.assertFalse(report['scientific_qualification'])
        self.assertFalse(report['original_nsyns_recovered'])
        self.assertEqual(report['production_episodes'],0)

    def test_association_uses_stored_edge_type_and_preserves_predicate_conflicts(self):
        node={key:dict(value=0.0,dtype='<f8') for key in ('x','z','tuning_angle')}
        witness=dict(name='known',source_id=5,target_id=7,raw_group=0,raw_offset=1,
            source_node=node,target_node=node,raw_weight=dict(value=10.0),raw_weight_dtype='<f8',
            allowed_nsyns=[3,4],source_population='e23Cux2',target_node_type_id=22,
            table_row={'edge_type_id':1,'syn_weight':1.0})
        row=dict(edge_type_id=8,source_query="pop_name=='e23Cux2'",target_query="node_type_id=='99'",
                 syn_weight=2.5,weight_sigma=30.0,delay=1.0,weight_function='test')
        edge=dict(global_row_id=20,source_node_id=5,target_node_id=7,edge_type_id=8,
            properties=dict(nsyns=dict(status='RECORDED',value=4),
                **{key:dict(status='MISSING') for key in ('syn_weight','weight_sigma','delay')}))
        case=dict(witness,matches_complete=True,matching_row_count=2,matching_rows=[edge,dict(edge,global_row_id=21)])
        rules={'test':lambda edges,source,target:np.asarray(edges['syn_weight']*edges['nsyns'])}
        result=arithmetic.associate_case(witness,case,{8:row},rules)
        self.assertEqual([r['global_row_id'] for r in result['source_rows']],[20,21])
        self.assertTrue(all(r['exact'] for r in result['source_rows']))
        self.assertTrue(all(not r['target_type_predicate_matches'] for r in result['source_rows']))
        self.assertEqual(result['prior_table_row']['syn_weight'],1.0)
        with self.assertRaisesRegex(ValueError,'COMPLETE_MATCHING'):
            arithmetic.associate_case(witness,dict(case,matches_complete=False),{8:row},rules)

    def test_changed_inputs_fail_before_unpickle(self):
        with tempfile.TemporaryDirectory(prefix='weight-arithmetic-binding-') as temporary:
            arguments,_,_=setup(Path(temporary))
            with arguments['types'].open('a') as stream:stream.write('changed\n')
            with mock.patch.object(arithmetic.pickle,'load') as unpickle:
                with self.assertRaisesRegex(ValueError,'BOUND_INPUT_DIFFERS_BEFORE_UNPICKLE'):
                    arithmetic.diagnose(**arguments)
                unpickle.assert_not_called()
            self.assertEqual(json.loads((arguments['output']/'report.json').read_text())['exit_code'],1)

    def test_missing_selector_does_not_substitute_a_case(self):
        with tempfile.TemporaryDirectory(prefix='weight-arithmetic-selection-') as temporary:
            arguments,_,_=setup(Path(temporary))
            arguments['config'].write_text(yaml.safe_dump({'cases':[dict(name='absent',selection='first',
                source_population='i23Pvalb',target_node_type_id=33)]}))
            with self.assertRaisesRegex(ValueError,'SELECTOR_HAS_NO_TUPLES'):
                arithmetic.diagnose(**arguments)


if __name__=='__main__':
    unittest.main()
