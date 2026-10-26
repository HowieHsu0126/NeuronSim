"""Verify development checkpoint intake and real search artifact delivery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from . import population_intervention_check as check
from . import population_intervention_replay as replay
from .population_intervention import ReferenceTarget, design_intervention
from .test_population_intervention import NativePulseEpisode, population_model


class DevelopmentInterventionTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((Path(__file__).parents[1] / 'configs/neural_population_intervention_development.yaml').read_text())
        self.design = dict(cases=[dict(case_id=f'train_block_{block:03d}_familiar_{template}',
            split='train', block=block, template_id=f'familiar_{template}',
            settings=dict(pulse_control=dict(initial_emission_ms=[30., 40., 50.])))
            for block in range(18) for template in range(4)])

    def test_registered_development_and_engineering_configs_remain_distinct(self):
        check.validate_development_config(self.config)
        engineering = yaml.safe_load((Path(__file__).parents[1] / 'configs/neural_population_intervention_check.yaml').read_text())
        check.validate_config(engineering)
        with self.assertRaises(ValueError):
            check.validate_config(self.config)
        for key, value in [('training_seeds',[41,42]),('reference_case_ids',self.config['reference_case_ids'][:-1]),
                           ('scientific_success_tolerance', .8),('joint_draws',4)]:
            changed = dict(self.config, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                check.validate_development_config(changed)

    def test_fitting_reference_and_selection_identities_reject_leakage(self):
        cases = check.development_cases(self.design, self.config)
        self.assertEqual(len(cases),72)
        for mutate in [lambda rows: rows[0].update(block=14),
                       lambda rows: rows[0].update(split='test'),
                       lambda rows: rows[0]['settings']['pulse_control'].update(initial_emission_ms=[31.,40.,50.]),
                       lambda rows: rows.append(deepcopy(rows[0]))]:
            changed=deepcopy(self.design);mutate(changed['cases'])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                check.development_cases(changed,self.config)

    def test_all_completed_seed_loaders_pass_before_search(self):
        model=population_model();native=NativePulseEpisode(model); calls=[]
        def load(training,*args):
            seed=int(training.parent.name.removeprefix('seed_'));calls.append(seed)
            binding=dict(seed=seed,config=dict(training_seeds=[41,42,43],fit_blocks=list(range(14)),validation_blocks=list(range(14,18))))
            return model,binding,f'digest{seed}'
        with patch.object(check,'_load_model',side_effect=load), patch.object(check,'sha256_file',return_value='receipt'):
            values=check.load_development_models(Path('/fixture'),Path('/acquisition'),Path('/fields'),native,self.config)
        self.assertEqual(calls,[41,42,43]);self.assertEqual([v[0] for v in values],[41,42,43])
        def fail(training,*args):
            if training.parent.name=='seed_42': raise ValueError('incomplete seed')
            return load(training,*args)
        with patch.object(check,'_load_model',side_effect=fail),patch.object(check,'sha256_file',return_value='receipt'):
            with self.assertRaisesRegex(ValueError,'incomplete seed'):
                check.load_development_models(Path('/fixture'),Path('/acquisition'),Path('/fields'),native,self.config)

    def test_wrong_seed_or_partition_is_rejected(self):
        for binding in [dict(seed=42,config=dict(training_seeds=[41,42,43],fit_blocks=list(range(14)),validation_blocks=list(range(14,18)))),
                        dict(seed=41,config=dict(training_seeds=[41,42,43],fit_blocks=list(range(15)),validation_blocks=list(range(15,18))))]:
            with self.subTest(binding=binding),patch.object(check,'_load_model',return_value=(None,binding,'checkpoint')):
                with self.assertRaisesRegex(ValueError,'registered three-seed'):
                    check.load_development_models(Path('/fixture'),None,None,None,self.config)

    def test_shared_reference_projection_uses_actual_field_denominators(self):
        model=population_model();native=NativePulseEpisode(model)
        native.source_identity={'source':'fixture'};native.binding={'episode':'fixture'}
        config=dict(reference_case_ids=['a','b'],decision_time_ms=20,horizon_ms=40)
        cases={name:dict(block=i) for i,name in enumerate(config['reference_case_ids'])}
        def source(*args,**kwargs):
            other=deepcopy(native)
            other.window=lambda start,stop:(other.counts[start//5:stop//5].copy(),other.stimulus[start//5:stop//5].copy())
            return other
        metadata=dict(field_ids=native.field_ids,unit_count=np.array([4,3]))
        with patch.object(check,'NativeRenewalEpisode',side_effect=source),patch.object(check,'fitting_scale',return_value=(None,np.array([10.,20.]),metadata)):
            target,bindings=check.reference_target(native,cases,config,Path('/acquisition'),Path('/fields'))
        expected=np.stack([native.counts[4:12,native.unit_field_index==f].sum(-1,dtype=np.float64) for f in range(2)],-1)/(np.array([4,3])*.005)
        np.testing.assert_array_equal(target.field_rate_hz,np.stack([expected,expected]))
        self.assertEqual(target.independent_block_ids,('0','1'));self.assertEqual(len(bindings),2)

    def test_real_search_delivery_preserves_every_candidate_and_engineering_schema(self):
        model=population_model();native=NativePulseEpisode(model)
        native.binding={'episode':'fixture'}
        config=dict(case_id='selected',reference_case_ids=['reference'],reference_role='fixture',
            decision_time_ms=20,horizon_ms=40,joint_draws=3,sampling_stream_seeds=[101,102,103],pulse_cost_weight=.25)
        cases=dict(selected=dict(settings=dict(pulse_control=dict(initial_emission_ms=[5.,17.,18.,20.,30.,40.,50.,60.]))))
        native=NativePulseEpisode(model,tuple(cases['selected']['settings']['pulse_control']['initial_emission_ms']))
        native.binding={'episode':'fixture'}
        graph=dict(resolved_source_network=dict(thalamic_input=True,thalamic_input_type='pulses',th_delay_pulse_generator=3.,th_pulse_start=8.,th_interval=10.),
                   source_config=dict(simulation=dict(sim_resolution=1.,t_presim=0.,t_sim=80.)))
        target=ReferenceTarget(np.zeros((1,8,2)),model.field_names,np.array([10.,20.]),('0',),.005)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'results'
            report=check.execute_search(model,native,cases,config,graph,Path('/acquisition'),Path('/fields'),output,
                dict(checkpoint_sha256='fixture'), 'ENGINEERING_NATIVE_PULSE_SEARCH_EXECUTED',(target,[]))
            self.assertEqual(report['status'],'ENGINEERING_NATIVE_PULSE_SEARCH_EXECUTED')
            self.assertEqual(len(report['candidates']),3)
            self.assertEqual(set(report['artifacts']),{'reference.npz','omit.npz','first_future_pulse.npz','original_future_pulses.npz','candidates.csv'})
            self.assertFalse(report['scientific_claim_qualified']);self.assertFalse(report['mechanistic_simulation_performed'])
            for name in ('omit','first_future_pulse','original_future_pulses'):
                with np.load(output/(name+'.npz')) as values:
                    self.assertEqual(values['unit_total_counts'].shape,(3,7))
                    self.assertEqual(values['terminal_ready_in_ticks'].shape,(3,7))
            self.assertEqual(json.loads((output/'analysis.json').read_text())['candidates'],report['candidates'])

    def test_complete_three_seed_development_delivery_and_missing_seed_rejection(self):
        model=population_model()
        initial=[5.,17.,18.,20.,30.,40.,50.,60.]
        native=NativePulseEpisode(model,tuple(initial));native.binding={'episode':'fixture'}
        native.source_identity=dict(native_poisson_rng_family='per_edge_philox4_32_10',graph_manifest_sha256='graph')
        native.duration_ticks=6000
        config=dict(self.config,expected_units=7,expected_fields=2,decision_time_ms=20,horizon_ms=40,joint_draws=3)
        cases={config['case_id']:dict(settings=dict(pulse_control=dict(initial_emission_ms=initial)))}
        target=ReferenceTarget(np.zeros((14,8,2)),model.field_names,np.array([10.,20.]),tuple(map(str,range(14))),.005)
        graph=dict(resolved_source_network=dict(thalamic_input=True,thalamic_input_type='pulses',th_delay_pulse_generator=3.,th_pulse_start=8.,th_interval=10.),
                   source_config=dict(simulation=dict(sim_resolution=1.,t_presim=0.,t_sim=6000.)))
        models=[(seed,model,{'seed':seed},'checkpoint','receipt') for seed in [41,42,43]]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config_path=root/'config.yaml';config_path.write_text(yaml.safe_dump(config))
            graph_path=root/'graph.json';graph_path.write_text(json.dumps(graph));(root/'design.json').write_text('{}')
            with patch.object(check,'validate_development_config'),patch.object(check,'development_cases',return_value=cases), \
                 patch.object(check,'NativeRenewalEpisode',return_value=native),patch.object(check,'sha256_file',return_value='graph'), \
                 patch.object(check,'reference_target',return_value=(target,[])),patch.object(check,'load_development_models',return_value=models):
                report=check.development(root,root,root,graph_path,config_path,root/'results')
                self.assertEqual(report['completed_seeds'],[41,42,43]);self.assertEqual(report['candidate_results'],9)
                self.assertFalse(report['independent_intervention_qualified'])
                for seed in [41,42,43]:
                    cell=json.loads((root/f'results/seed_{seed}/analysis.json').read_text())
                    self.assertEqual(cell['training_seed'],seed);self.assertEqual(len(cell['candidates']),3)
            # Bind the tiny generated fixture to real hashes before independent
            # replay; the model/native loader alone substitutes source dimensions.
            from .provenance import sha256_file
            digest=sha256_file(graph_path)
            native.source_identity['graph_manifest_sha256']=digest
            aggregate=json.loads((root/'results/analysis.json').read_text())
            for seed,declared in zip([41,42,43],aggregate['seeds']):
                cell_path=root/f'results/seed_{seed}/analysis.json';cell=json.loads(cell_path.read_text())
                cell['graph_manifest_sha256']=digest;cell['training_receipt_sha256']='receipt'
                cell['artifacts']={name:sha256_file(cell_path.parent/name) for name in cell['artifacts']}
                cell_path.write_text(json.dumps(cell))
                declared['analysis_sha256']=sha256_file(cell_path)
            (root/'results/analysis.json').write_text(json.dumps(aggregate))
            def source_load(training,*args):
                seed=int(training.parent.name.removeprefix('seed_'))
                return model,{'seed':seed},'checkpoint'
            def source_hash(path):
                return 'receipt' if str(path).endswith('/training/run.json') else sha256_file(path)
            with patch.object(replay,'validate_development_config'),patch.object(replay,'development_cases',return_value=cases), \
                 patch.object(replay,'NativeRenewalEpisode',return_value=native),patch.object(replay,'sha256_file',side_effect=source_hash), \
                 patch.object(replay,'native_references',return_value=(dict(field_rate_hz=target.field_rate_hz,scale_hz=target.scale_hz,
                     field_ids=np.asarray(target.field_names),block_ids=np.asarray(target.independent_block_ids)),[])), \
                 patch.object(replay,'_load_model',side_effect=source_load):
                replayed=replay.replay_development(root,root,root,graph_path,config_path,root/'results',root/'replay.json')
                self.assertEqual(replayed['completed_seeds'],[41,42,43])
                self.assertTrue(all(row['saved_all_unit_totals_exact'] for row in replayed['seeds']))
                self.assertFalse(replayed['per_bin_full_unit_counts_exact_replay_claimed'])
                (root/'results/seed_42/analysis.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'seed analysis changed'):
                    replay.replay_development(root,root,root,graph_path,config_path,root/'results',root/'tampered_replay.json')
                self.assertFalse((root/'tampered_replay.json').exists())
            native.source_identity['graph_manifest_sha256']='graph'
            with patch.object(check,'validate_development_config'),patch.object(check,'development_cases',return_value=cases), \
                 patch.object(check,'NativeRenewalEpisode',return_value=native),patch.object(check,'sha256_file',return_value='graph'), \
                 patch.object(check,'reference_target',return_value=(target,[])),patch.object(check,'load_development_models',side_effect=ValueError('missing seed')):
                with self.assertRaisesRegex(ValueError,'missing seed'):
                    check.development(root,root,root,graph_path,config_path,root/'failed_results')
                self.assertFalse((root/'failed_results').exists())

    def test_real_cli_rejects_checkpoint_without_receipt(self):
        command=[sys.executable,'-m','population_dynamics.population_intervention_check']
        for name in ('acquisition','fields','graph-manifest','config','output','checkpoint'):
            command.extend(('--'+name,'/missing'))
        result=subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--checkpoint requires --model-receipt',result.stderr)


if __name__=='__main__':
    unittest.main()
