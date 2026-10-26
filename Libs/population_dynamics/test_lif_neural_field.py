"""Independent integral and source-edge checks for the mean-field comparator."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx

from .lif_neural_field import ALPHA_VOLTAGE_SHIFT, LIFNeuralField, build_projection, compare_population_campaign, evaluate_campaign, forecast_episode, lif_rate, projected_edge_moments, sampled_mean_field_mse, validate_campaign_config
from .provenance import sha256_file
from .spatial_connectivity import EDGE_DTYPE


class LIFNeuralFieldTests(unittest.TestCase):
    def test_siegert_matches_independent_adaptive_integral_and_deterministic_limit(self):
        args=dict(reset_mv=0.,threshold_mv=15.,tau_m_seconds=.01,refractory_seconds=.0021,tau_syn_seconds=.0005)
        for mu,sigma in [(8.,3.),(15.,3.),(25.,5.),(-20.,5.),(20.,0.),(10.,0.)]:
            with self.subTest(mu=mu,sigma=sigma):
                if sigma:
                    shift=ALPHA_VOLTAGE_SHIFT/2*np.sqrt(.05)
                    integral=quad(lambda x:erfcx(-x),(0-mu)/sigma+shift,(15-mu)/sigma+shift,epsabs=1e-9,epsrel=1e-11)[0]
                    expected=1/(.0021+.01*np.sqrt(np.pi)*integral)
                else:expected=0. if mu<=15 else 1/(.0021+.01*np.log(mu/(mu-15)))
                np.testing.assert_allclose(lif_rate(mu,sigma,**args),expected,rtol=1e-7,atol=1e-9)

    def test_duplicate_weights_and_tc_mask_match_explicit_edge_oracle(self):
        ids=np.array([1,2,3,4,5]);groups=np.array([0,0,1,-1,-1]);sizes=np.array([2,1]);mask=np.array([False,False,False,True,False])
        edges=np.array([(1,3,2.,.5,0,0),(1,3,3.,.5,1,0),(3,1,-4.,.7,0,0),(4,1,6.,.5,0,0),(5,1,9.,.5,0,0)],dtype=EDGE_DTYPE)
        actual=projected_edge_moments(edges,ids,groups,sizes,mask,.1)
        np.testing.assert_array_equal(actual[5][0],[[0,0],[5,0]])
        np.testing.assert_array_equal(actual[5][1],[[0,0],[25,0]])
        np.testing.assert_array_equal(actual[5][2],[3,0])
        np.testing.assert_array_equal(actual[7][0],[[0,-2],[0,0]])
        np.testing.assert_array_equal(actual[7][1],[[0,8],[0,0]])
        changed=edges.copy();changed['raw_delay'][1]=.6
        diagnostics={}
        split=projected_edge_moments(changed,ids,groups,sizes,mask,.1,diagnostics=diagnostics)
        self.assertEqual(diagnostics,dict(cross_delay_source_target_pairs=1,cross_delay_edges=2))
        self.assertEqual(split[5][0][1,0]+split[6][0][1,0],5)
        self.assertEqual(split[5][1][1,0]+split[6][1][1,0],13)
        changed=edges.copy();changed['source'][0]=999
        with self.assertRaisesRegex(ValueError,'identity'):projected_edge_moments(changed,ids,groups,sizes,mask,.1)

    def test_mean_field_exact_deterministic_constant_drive_and_future_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=dict(V_reset=-65.,V_th=-50.,E_L=-65.,C_m=250.,tau_m=10.,tau_syn_ex=.5,tau_syn_in=.5,t_ref=2.)
            np.savez(root/'projection.npz',delay_ticks=np.array([5]),mean_weights_pa=np.zeros((1,1,1)),
                squared_weights_pa2=np.zeros((1,1,1)),tc_impulses_pa=np.array([[200.]]),field_ids=np.array(['a']),
                unit_ids=np.array([1,2]),unit_field_index=np.array([0,0]),unit_count=np.array([2]),field_populations=np.array(['L23E']))
            net=dict(populations=['L23E'],bg_rate=0.,ext_indegrees=[0],DC_amp=[500.],weight_ext=80.,delay_poisson=1.5)
            (root/'projection.json').write_text(json.dumps(dict(status='PROJECTED_COMPLETE_SOURCE_GRAPH',projection_sha256=sha256_file(root/'projection.npz'),parameters=p,network=net,dt_ms=.1)))
            model=LIFNeuralField(root,1.)
            h=np.zeros((20,1),dtype=int);args=dict(boundary_tick=1000,horizon_ticks=500,bin_ticks=50)
            counts,proof=model.forecast(h,np.array([],dtype=int),**args)
            target=1/(.0021+.01*np.log(20/5));a=np.exp(-.0001/.001)
            tick_rates=target*(1-a**np.arange(500))
            expected=(tick_rates*.0001*2).reshape(10,50).sum(1)
            np.testing.assert_allclose(counts[:,0],expected,rtol=1e-12,atol=1e-12)
            controlled,_=model.forecast(h,np.array([1100]),**args)
            np.testing.assert_array_equal(controlled[:2],counts[:2])
            self.assertGreater(controlled[3:].sum(),counts[3:].sum())
            self.assertFalse(proof['unit_outputs_available']);self.assertFalse(proof['distribution_scores_available'])
            self.assertIsNone(proof['transfer_quadrature_max_difference_hz'])
            self.assertEqual(proof['transfer_quadrature_checks'],0)

    def test_small_diffusion_and_zero_pre_episode_history(self):
        args=dict(reset_mv=0.,threshold_mv=15.,tau_m_seconds=.01,refractory_seconds=.0021,tau_syn_seconds=.0005)
        self.assertEqual(float(lif_rate(8.,1e-12,**args)),0.)
        expected=float(lif_rate(20.,0.,**args))
        self.assertAlmostEqual(float(lif_rate(20.,1e-12,**args)),expected,places=8)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            np.savez(root/'projection.npz',delay_ticks=np.array([100]),mean_weights_pa=np.array([[[1.]]]),
                squared_weights_pa2=np.array([[[1.]]]),tc_impulses_pa=np.zeros((1,1)),field_ids=np.array(['a']),
                unit_ids=np.array([1]),unit_field_index=np.array([0]),unit_count=np.array([1]),field_populations=np.array(['L23E']))
            parameters=dict(V_reset=-65.,V_th=-50.,E_L=-65.,C_m=250.,tau_m=10.,tau_syn_ex=.5,tau_syn_in=.5,t_ref=2.)
            net=dict(populations=['L23E'],bg_rate=0.,ext_indegrees=[0],DC_amp=[0.],weight_ext=80.,delay_poisson=1.5)
            (root/'projection.json').write_text(json.dumps(dict(status='PROJECTED_COMPLETE_SOURCE_GRAPH',
                projection_sha256=sha256_file(root/'projection.npz'),parameters=parameters,network=net,dt_ms=.1)))
            model=LIFNeuralField(root,1.)
            counts,_=model.forecast(np.array([[1]]),np.array([],dtype=int),boundary_tick=50,horizon_ticks=50,bin_ticks=50)
            # No observed event can arrive before tick100; the prefix rate
            # relaxes toward zero without fabricated emissions before zero.
            expected=(200*np.exp(-np.arange(50)*.1)*.0001).sum()
            self.assertAlmostEqual(float(counts.sum()),expected,places=12)

    def test_field_mean_score_matches_independent_distinct_draw_products(self):
        samples=np.array([[[0,2],[3,1]],[[2,4],[5,3]],[[1,3],[4,2]]])
        truth=np.array([[1,2],[4,1]]);sd=np.array([2.,3.]);units=np.array([2,4]);denominator=sd*units*.005
        products=[]
        for first in range(3):
            for second in range(3):
                if first!=second:products.append(((samples[first]-truth)*(samples[second]-truth)/denominator**2).mean())
        self.assertAlmostEqual(sampled_mean_field_mse(samples,truth,sd,units,5.),np.mean(products),places=8)
        opposite=np.array([[[0]],[[2]]]);actual=np.array([[1]])
        self.assertLess(sampled_mean_field_mse(opposite,actual,np.ones(1),np.ones(1),1000),0)

    def test_source_graph_cli_path_and_exact_forecast_replay(self):
        from .spatial_field import PeriodicPopulationGrid
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);graph=root/'graph';fields=root/'fields';episode=root/'episode'
            for p in (graph,fields,episode):p.mkdir()
            # One source neuron in each of the sixteen fixed spatial cells.
            xy=np.array([(x,y) for y in (-.375,-.125,.125,.375) for x in (-.375,-.125,.125,.375)])
            ids=np.arange(1,18);labels=np.array(['L23E']*16+['TC']);positions=np.vstack([xy,[[0,0]]])
            np.savez(graph/'layout.npz',unit_ids=ids,populations=labels,positions_mm=positions)
            edges=np.array([(17,1,40.,.5,0,0),(1,2,2.,.5,0,0)],dtype=EDGE_DTYPE)
            np.save(graph/'edges.npy',edges)
            parameters=dict(V_reset=-65.,V_th=-50.,E_L=-65.,C_m=250.,tau_m=10.,tau_syn_ex=.5,tau_syn_in=.5,t_ref=2.)
            net=dict(neuron_model='iaf_psc_exp',dc_input=False,thalamic_input=True,thalamic_input_type='pulses',extent=1.,th_radius=.5,
                neuron_params=parameters,populations=['L23E','TC'],bg_rate=8.,ext_indegrees=[1600],DC_amp=[0.],weight_ext=80.,delay_poisson=1.5,
                th_delay_pulse_generator=1.,th_pulse_start=50.,th_interval=50.)
            manifest=dict(schema='mesocircuit_spatial_projection_graph_v2',status='GENERATED',external_device_edges_included=False,
                resolved_source_network=net,resolved_simulation=dict(sim_resolution=.1),layout_sha256=sha256_file(graph/'layout.npz'),
                parts=[dict(path='edges.npy',edges=2,source_population='mixed',target_population='L23E',sha256=sha256_file(graph/'edges.npy'))],edges=2)
            (graph/'manifest.json').write_text(json.dumps(manifest))
            grid=PeriodicPopulationGrid.from_positions(ids[:16],labels[:16],xy,extent_mm=1.,cells_per_axis=4,definition='fixture')
            np.savez(fields/'spatial_fields.npz',source_unit_ids=ids[:16],unit_field_index=grid.mapping.group_index,
                field_ids=grid.mapping.field_ids,total_unit_count=np.ones(16,dtype=int),count_sum=np.zeros((40,16),dtype=int),bin_edges_ticks=np.arange(41)*50)
            native=dict(status='EXECUTED',graph_manifest_sha256=sha256_file(graph/'manifest.json'),dt_ms=.1,duration_ms=200.,
                config=dict(pulse_control=dict(initial_emission_ms=[120.])))
            (episode/'report.json').write_text(json.dumps(native))
            (fields/'summary.json').write_text(json.dumps(dict(episode_report_sha256=sha256_file(episode/'report.json'),artifacts={'spatial_fields.npz':sha256_file(fields/'spatial_fields.npz')})))
            report=build_projection(graph,fields,episode,root/'projection')
            self.assertEqual(report['source_edges'],2);self.assertEqual(report['source_units'],16)
            first=forecast_episode(root/'projection',episode,fields,root/'forecast',relaxation_ms=5.,origin_ms=100.,horizon_ms=50.)
            replay=forecast_episode(root/'projection',episode,fields,root/'forecast',relaxation_ms=5.,origin_ms=100.,horizon_ms=50.,replay=True)
            self.assertEqual(first['status'],'EXECUTED_CONDITIONAL_MEAN_FIELD')
            self.assertEqual(replay['status'],'PASS_EXACT_MEAN_FIELD_FRESH_PROCESS_REPLAY')
            self.assertGreater(first['predicted_spikes_per_unit'],0)
            # The CLI replay must execute in a separate interpreter.
            (root/'forecast/replay.json').unlink()
            child=subprocess.run([sys.executable,'-m','Libs.population_dynamics.lif_neural_field',
                '--projection',str(root/'projection'),'--episode',str(episode),'--fields',str(fields),
                '--output',str(root/'forecast'),'--relaxation-ms','5','--origin-ms','100','--horizon-ms','50','--replay'],
                capture_output=True,text=True)
            self.assertEqual(child.returncode,0,child.stdout+child.stderr)
            self.assertEqual(json.loads((root/'forecast/replay.json').read_text())['status'],'PASS_EXACT_MEAN_FIELD_FRESH_PROCESS_REPLAY')
            # Exercise launcher staging with native-format graph/field inputs.
            # Source execution placement remains the caller's Slurm boundary.
            from Libs.scripts.run_lif_neural_field import stage
            repo=Path(__file__).resolve().parents[2]
            base=root/'inputs_source';case='train_block_000_familiar_0'
            (base/'train').mkdir(parents=True);(base/'train_fields').mkdir()
            shutil.copytree(episode,base/'train'/case);shutil.copytree(fields,base/'train_fields'/case)
            work=root/'staged';work.mkdir()
            stage(repo,work,dict(source_graph=str(graph),cohort=str(base),diagnostic_case=case))
            for arguments in [
                ['--graph','graph','--output','projection'],
                ['--projection','projection','--output','forecast','--relaxation-ms','5','--origin-ms','100','--horizon-ms','50'],
                ['--projection','projection','--output','forecast','--relaxation-ms','5','--origin-ms','100','--horizon-ms','50','--replay']]:
                child=subprocess.run([sys.executable,'-m','Libs.population_dynamics.lif_neural_field',
                    '--episode','inputs/train/'+case,'--fields','inputs/train_fields/'+case,*arguments],cwd=work,capture_output=True,text=True)
                self.assertEqual(child.returncode,0,child.stdout+child.stderr)
            # A rebound geometry mismatch must still be rejected by the field map.
            changed=np.zeros_like(positions)
            np.savez(graph/'layout.npz',unit_ids=ids,populations=labels,positions_mm=changed)
            manifest['layout_sha256']=sha256_file(graph/'layout.npz');(graph/'manifest.json').write_text(json.dumps(manifest))
            native['graph_manifest_sha256']=sha256_file(graph/'manifest.json');(episode/'report.json').write_text(json.dumps(native))
            summary=json.loads((fields/'summary.json').read_text());summary['episode_report_sha256']=sha256_file(episode/'report.json');(fields/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'observation map'):build_projection(graph,fields,episode,root/'corrupted')

    def test_campaign_keeps_all_candidates_and_uses_saved_fit_scale(self):
        """The real selection entry point runs every condition on a tiny graph."""
        import csv
        from .spatial_field import PeriodicPopulationGrid
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cohort=root/'cohort';graph=root/'graph';graph.mkdir();(cohort/'train').mkdir(parents=True)
            xy=np.array([(x,y) for y in (-.375,-.125,.125,.375) for x in (-.375,-.125,.125,.375)])
            ids=np.arange(1,18);labels=np.array(['L23E']*16+['TC']);positions=np.vstack([xy,[[0,0]]])
            np.savez(graph/'layout.npz',unit_ids=ids,populations=labels,positions_mm=positions)
            edges=np.array([(17,1,40.,5.,0,0)],dtype=EDGE_DTYPE);np.save(graph/'edges.npy',edges)
            parameters=dict(V_reset=-65.,V_th=-50.,E_L=-65.,C_m=250.,tau_m=10.,tau_syn_ex=.5,tau_syn_in=.5,t_ref=2.)
            net=dict(neuron_model='iaf_psc_exp',dc_input=False,thalamic_input=True,thalamic_input_type='pulses',extent=1.,th_radius=.5,
                neuron_params=parameters,populations=['L23E','TC'],bg_rate=0.,ext_indegrees=[0],DC_amp=[500.],weight_ext=80.,delay_poisson=2.,
                th_delay_pulse_generator=1.,th_pulse_start=50.,th_interval=50.)
            manifest=dict(schema='mesocircuit_spatial_projection_graph_v2',status='GENERATED',external_device_edges_included=False,
                resolved_source_network=net,resolved_simulation=dict(sim_resolution=1.),layout_sha256=sha256_file(graph/'layout.npz'),
                parts=[dict(path='edges.npy',edges=1,source_population='TC',target_population='L23E',sha256=sha256_file(graph/'edges.npy'))],edges=1)
            (graph/'manifest.json').write_text(json.dumps(manifest))
            grid=PeriodicPopulationGrid.from_positions(ids[:16],labels[:16],xy,extent_mm=1.,cells_per_axis=4,definition='fixture')
            cases=[dict(case_id=f'train_block_{block:03}_familiar_{condition}',block=block,split='train',family='trajectory',template_id=str(condition))
                for block in range(18) for condition in range(4)]
            (cohort/'train/design.json').write_text(json.dumps(dict(acquired_split='train',cases=cases)))
            for case in cases:
                if case['block']<14:continue
                episode=cohort/'train'/case['case_id'];fields=cohort/'train_fields'/case['case_id'];episode.mkdir();fields.mkdir(parents=True)
                np.savez(fields/'spatial_fields.npz',source_unit_ids=ids[:16],unit_field_index=grid.mapping.group_index,
                    field_ids=grid.mapping.field_ids,total_unit_count=np.ones(16,dtype=int),count_sum=np.zeros((1200,16),dtype=int),bin_edges_ticks=np.arange(1201)*5)
                (episode/'report.json').write_text(json.dumps(dict(status='EXECUTED',graph_manifest_sha256=sha256_file(graph/'manifest.json'),
                    dt_ms=1.,duration_ms=6000.,config=dict(pulse_control=dict(initial_emission_ms=[1200.])))))
                (fields/'summary.json').write_text(json.dumps(dict(episode_report_sha256=sha256_file(episode/'report.json'),
                    artifacts={'spatial_fields.npz':sha256_file(fields/'spatial_fields.npz')})))
            build_projection(graph,fields,episode,root/'projection')
            scale=root/'seed_41/forecasts/fitting_field_scale.npz';scale.parent.mkdir(parents=True)
            np.savez(scale,field_ids=grid.mapping.field_ids,unit_count=np.ones(16,dtype=int),standard_deviation_hz=np.arange(1,17,dtype=float))
            (scale.parents[1]/'delivery.json').write_text(json.dumps(dict(status='PASS',training_seed=41,
                files=[dict(path='forecasts/fitting_field_scale.npz',sha256=sha256_file(scale))])))
            config=dict(schema='source_projected_lif_mean_field_development_v1',fit_blocks=list(range(14)),validation_blocks=list(range(14,18)),
                bin_ms=5.,origins_ms=[1000.,3000.,5000.],horizons_ms=[20.,80.,500.],relaxation_candidates_ms=[1.,5.,10.],
                selection_split='validation',selection_metric='fitting_standardized_field_rate_mse',formal_evaluation=False)
            result=evaluate_campaign(root/'projection',cohort,scale,config,root/'evaluation')
            self.assertEqual(result['score_rows'],432)
            self.assertEqual(result['validation_episodes'],16)
            rows=list(csv.DictReader((root/'evaluation/candidate_scores.csv').open()))
            self.assertEqual({int(row['block']) for row in rows},{14,15,16,17})
            means={candidate:np.mean([float(row['fitting_standardized_field_rate_mse']) for row in rows if float(row['relaxation_ms'])==candidate])
                for candidate in [1.,5.,10.]}
            self.assertEqual(result['selected_relaxation_ms'],min(means,key=means.get))
            row=rows[0]
            with np.load(root/'evaluation'/row['forecast_artifact']) as arrays:
                independent=np.mean((arrays['expected_field_hz'][:4]/np.arange(1,17))**2)
            self.assertAlmostEqual(float(row['fitting_standardized_field_rate_mse']),independent,places=12)
            self.assertFalse(result['scientific_claim_qualified'])
            with self.assertRaisesRegex(ValueError,'protocol'):validate_campaign_config(dict(config,selection_split='test'))
            from Libs.scripts.run_lif_neural_field import stage
            diagnostic=root/'diagnostic';diagnostic.mkdir();shutil.copytree(root/'projection',diagnostic/'projection')
            (diagnostic/'delivery.json').write_text(json.dumps(dict(status='COMPLETE_SOURCE_GRAPH_AND_NATIVE_PREFIX_MEAN_FIELD_REPLAY',
                exit_code=0,files=[dict(path='projection/'+name,sha256=sha256_file(diagnostic/'projection'/name))
                for name in ['projection.json','projection.npz']])))
            inputs=[p for p in cohort.rglob('*') if p.is_file()]
            (cohort/'delivery.json').write_text(json.dumps(dict(status='VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE',completed_episodes=72,
                cortical_units=100000,files=[dict(path=str(p.relative_to(cohort)),sha256=sha256_file(p)) for p in inputs])))
            work=root/'staged_campaign';work.mkdir()
            stage(Path(__file__).resolve().parents[2],work,dict(config,cohort=str(cohort),fitting_scale=str(scale)),
                mode='campaign',projection_source=diagnostic)
            self.assertEqual(sha256_file(work/'fitting_field_scale.npz'),sha256_file(scale))
            self.assertTrue((work/'inputs/train'/cases[-1]['case_id']/'report.json').is_file())
            selection=root/'selection';selection.mkdir();shutil.copytree(root/'evaluation',selection/'evaluation')
            shutil.copy2(scale,selection/'fitting_field_scale.npz')
            paths=[p for p in selection.rglob('*') if p.is_file()]
            (selection/'delivery.json').write_text(json.dumps(dict(status='COMPLETE_DESCRIPTIVE_DEVELOPMENT_MEAN_FIELD_SELECTION',exit_code=0,
                files=[dict(path=str(p.relative_to(selection)),sha256=sha256_file(p)) for p in paths])))
            campaign=root/'population'
            for seed in [41,42,43]:
                cell=campaign/f'seed_{seed}';forecast=cell/'forecasts';forecast.mkdir(parents=True)
                shutil.copy2(scale,forecast/'fitting_field_scale.npz')
                for case in cases:
                    if case['block']<14:continue
                    for origin in [1000,3000,5000]:
                        np.savez(forecast/f"{case['case_id']}_origin_{origin}.npz",
                            sampled_field_counts=np.ones((16,100,16),dtype=int),observed_field_counts=np.zeros((100,16),dtype=int))
                evaluation=dict(status='EVALUATED_DEVELOPMENT',training_seed=seed,formal_test_accessed=False,
                    acquisition_design_sha256=result['binding']['cohort_design_sha256'],
                    configuration=dict(origins_ms=config['origins_ms'],horizons_ms=config['horizons_ms'],joint_samples=16))
                (forecast/'evaluation.json').write_text(json.dumps(evaluation))
                (cell/'replay.json').write_text(json.dumps(dict(status='PASS',training_seed=seed,verified_origins=48,
                    all_origin_fields_and_latents_exact=True,evaluation_sha256=sha256_file(forecast/'evaluation.json'))))
                paths=[p for p in cell.rglob('*') if p.is_file()]
                (cell/'delivery.json').write_text(json.dumps(dict(status='PASS',exit_code=0,training_seed=seed,
                    files=[dict(path=str(p.relative_to(cell)),sha256=sha256_file(p)) for p in paths])))
            comparison=compare_population_campaign(selection,campaign,root/'comparison')
            self.assertEqual(comparison['score_rows'],432)
            comparison_rows=list(csv.DictReader((root/'comparison/mean_field_comparison.csv').open()))
            expected=np.mean((np.ones((4,16))/(np.arange(1,17)*.005))**2)
            self.assertAlmostEqual(float(comparison_rows[0]['population_mean_field_mse_corrected']),expected,places=10)
            self.assertFalse(comparison['samples_regenerated'])
            from Libs.scripts.run_lif_neural_field import run
            from types import SimpleNamespace
            from unittest.mock import patch
            runner_work=root/'comparison_work';runner_work.mkdir()
            with patch.dict('os.environ',dict(SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_ID='fixture',SLURM_JOB_GPUS='')):
                status=run(SimpleNamespace(repo=Path(__file__).resolve().parents[2],work=runner_work,mode='comparison',
                    projection_source=None,selection_source=selection,population_campaign=campaign))
            self.assertEqual(status,0)
            delivered=Path(__file__).resolve().parents[2]/'Output/neural_population_dynamics/baselines/lif_neural_field/run_fixture/delivery.json'
            self.assertEqual(json.loads(delivered.read_text())['status'],'COMPLETE_DESCRIPTIVE_COMMON_MEAN_FIELD_COMPARISON')
            shutil.rmtree(delivered.parent)
            path=campaign/'seed_43/forecasts/train_block_014_familiar_0_origin_1000.npz'
            np.savez(path,sampled_field_counts=np.zeros((16,100,16),dtype=int),observed_field_counts=np.zeros((100,16),dtype=int))
            with self.assertRaisesRegex(ValueError,'samples changed'):compare_population_campaign(selection,campaign,root/'corrupted_comparison')


if __name__=='__main__':unittest.main()
