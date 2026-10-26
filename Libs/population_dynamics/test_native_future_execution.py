"""Verify complete-future admission, staging and serialized execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_nest_gpu_live_control as runner


class FutureExecutionTests(unittest.TestCase):
    def config(self):
        return yaml.safe_load((Path(__file__).parents[1]/'configs/neural_population_native_future_ensemble.yaml').read_text())

    def proof(self):
        return dict(status='PASS', exit_code=0, native_library_sha256='a'*64,
            complete_native_executions=4, boundary_tick=10000, reset_tick=10015,
            first_resampled_arrival_tick=10016, original_complete_episode_regression_exact=True,
            same_future_seed_full_array_replay_exact=True, full100000_unit_boundary_states_exact=True,
            committed_event_prefix_exact=True, distinct_future_seeds_change_source_events=True,
            plan=dict(case_id=runner.COUPLED_CASE, core_neurons=100000, duration_ms=6000))

    def test_future_qualification_rejects_partial_or_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); proof=self.proof()
            (root/'delivery.json').write_text(json.dumps(proof))
            self.assertEqual(runner.require_future_qualification(root, 'a'*64), proof)
            for key in ('status', 'exit_code', 'native_library_sha256', 'complete_native_executions',
                        'committed_event_prefix_exact', 'full100000_unit_boundary_states_exact',
                        'same_future_seed_full_array_replay_exact', 'first_resampled_arrival_tick'):
                changed=deepcopy(proof); changed[key]='invalid'
                (root/'delivery.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.require_future_qualification(root, 'a'*64)

    def test_future_vectors_preserve_draws_and_independent_streams(self):
        config=self.config(); seeds=runner.future_ensemble_seeds(config)
        self.assertEqual(len(seeds), 16)
        self.assertEqual(len({v for row in seeds for v in row}), 128)
        for key,value in [('joint_samples',4),('future_background_seed_stride',7),
                          ('future_background_seed_base',True),('future_background_seed_base',0),
                          ('future_background_seed_base',2**31-100),('formal_evaluation',True)]:
            changed=dict(config);changed[key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                runner.future_ensemble_seeds(changed)

    def test_cli_requires_full_qualification_before_execution(self):
        argv=['runner','--repo','repo','--work','work','--output','output',
              '--future-poisson-rng-ensemble','--native-build','build','--producer-software','software']
        with mock.patch.object(runner.sys,'argv',argv), mock.patch.object(runner,'execute_future_ensemble') as execute:
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code,2);execute.assert_not_called()
        argv += ['--coupled-qualification','qualified']
        with mock.patch.object(runner.sys,'argv',argv), mock.patch.object(runner,'execute_future_ensemble',return_value=0) as execute:
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code,0)
            execute.assert_called_once_with(Path('repo'),Path('work'),Path('output'),
                                            Path('build'),Path('qualified'),Path('software'))

    def test_complete_execution_serializes_all_draws_and_measures_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';target=root/'output'
            seeds=runner.future_ensemble_seeds(self.config())
            plan=dict(native_library_sha256='a'*64, graph_bytes=1,
                      settings=dict(state_observation_ticks=[]))
            manifest={}; archive_records=[]; commands=[]
            def stage(*args):
                (work/'inputs').mkdir()
                (work/'inputs/design.json').write_text(json.dumps(dict(cases=[dict(case_id=runner.COUPLED_CASE,
                    settings=dict(pulse_control={'original':'program'}))])))
                return plan,manifest
            def command(args,**kwargs):
                commands.append(args)
                if 'Libs.population_dynamics.nest_gpu_spatial' in args:
                    output=Path(args[args.index('--output')+1]); output.mkdir()
                    (output/'report.json').write_text(json.dumps(dict(status='EXECUTED',elapsed_seconds=2.,
                        import_seconds=.5,simulation_seconds=1.,calibration_seconds=.1,peak_process_rss_mib=3.)))
                elif 'Libs.population_dynamics.native_future_ensemble' in args:
                    output=Path(args[args.index('--output')+1]);output.mkdir()
                    (output/'analysis.json').write_text(json.dumps(dict(status='DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT',
                        draws=16,generated_unit_count_values=160000000,full_unit_counts_saved=True,
                        all_committed_events_and_states_exact=True,all_tc_programs_exact=True,
                        branch_source_identity=dict(native_library_sha256='a'*64))))
                return mock.Mock(returncode=0)
            def archive(source,destination,record):
                destination.mkdir();archive_records.append(record)
            env=dict(SLURMD_NODENAME='gpu03',SLURM_JOB_GPUS='0',SLURM_JOB_ID='123456')
            with mock.patch.dict(runner.os.environ,env), mock.patch.object(runner,'stage_coupled_inputs',side_effect=stage), \
                 mock.patch.object(runner,'stage_future_ensemble_inputs',return_value=(seeds,{})), \
                 mock.patch.object(runner,'stage_native_graph'), mock.patch.object(runner.shutil,'disk_usage',return_value=mock.Mock(free=10**12)), \
                 mock.patch.object(runner.shutil,'rmtree'), mock.patch.object(runner.subprocess,'run',side_effect=command), \
                 mock.patch.object(runner,'archive',side_effect=archive):
                code=runner.execute_future_ensemble(root,work,target,root/'build',root/'qualified',root/'software')
            self.assertEqual(code,0)
            native=[c for c in commands if 'Libs.population_dynamics.nest_gpu_spatial' in c]
            self.assertEqual(len(native),16)
            self.assertEqual(sum('Libs.population_dynamics.native_future_ensemble' in c for c in commands),1)
            for index,c in enumerate(native):
                config=yaml.safe_load(Path(c[c.index('--config')+1]).read_text())
                self.assertEqual(config['future_background_noise']['seeds_per_population'],seeds[index])
                self.assertFalse(config['live_pulse_control'])
                self.assertEqual(config['pulse_control'],{'original':'program'})
            health=[c for c in commands if 'Libs.population_dynamics.resource_probe' in c]
            self.assertEqual(len(health),17)
            self.assertTrue(all(c[:3]==[sys.executable,'-m','Libs.population_dynamics.resource_probe'] for c in health))
            result=json.loads((target/'delivery.json').read_text())
            self.assertEqual(result['complete_native_executions'],16)
            self.assertEqual(result['simulated_biological_ms'],96000)
            self.assertEqual(result['target_output_biological_ms'],8000)
            self.assertEqual(result['generated_unit_count_values'],160000000)
            self.assertFalse(result['matched_fidelity_qualified'])
            self.assertGreater(result['task_through_archive_seconds'],result['pipeline_before_archive_seconds'])


if __name__=='__main__':
    unittest.main()
