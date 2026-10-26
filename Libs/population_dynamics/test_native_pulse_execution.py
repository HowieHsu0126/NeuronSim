"""Verify native execution gates and manifest-bound graph staging."""
from __future__ import annotations

from copy import deepcopy
from unittest import mock
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_nest_gpu_live_control as runner


class NativePulseExecutionTests(unittest.TestCase):
    def test_all_native_gates_and_exact_coupled_comparisons_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);canary=root/'canary';coupled=root/'coupled'
            (canary/'results').mkdir(parents=True);coupled.mkdir()
            proof=dict(status='PASS',exit_code=0,native_library_sha256='a'*64)
            checks=dict(status='PASS',all_six_cases_executed=True,no_mutation_original_library_regression_verified=True)
            full=dict(status='PASS',exit_code=0,plan=dict(native_library_sha256='a'*64,core_neurons=100000,duration_ms=6000,case_id=runner.COUPLED_CASE))
            comparison=dict(status='PASS',comparisons=[dict(exact=True)])
            def save(a,b,c,d):
                for path,value in [(canary/'delivery.json',a),(canary/'results/report.json',b),(coupled/'delivery.json',c),(coupled/'comparison.json',d)]:
                    path.write_text(json.dumps(value))
            save(proof,checks,full,comparison)
            runner.require_native_qualification(canary,coupled,'a'*64)
            for change in ['failed_canary','incomplete_cases','missing_original','changed_library','failed_coupled','not_exact','empty_comparison']:
                a,b,c,d=map(deepcopy,(proof,checks,full,comparison))
                if change=='failed_canary': a['exit_code']=2
                elif change=='incomplete_cases': b['all_six_cases_executed']=False
                elif change=='missing_original': b['no_mutation_original_library_regression_verified']=False
                elif change=='changed_library': c['plan']['native_library_sha256']='b'*64
                elif change=='failed_coupled': c['status']='FAILED'
                elif change=='not_exact': d['comparisons'][0]['exact']=False
                else: d['comparisons']=[]
                save(a,b,c,d)
                with self.subTest(change=change),self.assertRaises(ValueError):
                    runner.require_native_qualification(canary,coupled,'a'*64)

    def test_graph_staging_binds_layout_and_all_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/runner.GRAPH;source.mkdir(parents=True)
            work=root/'work';work.mkdir();(work/'graph_manifest.json').write_text('{"bound":"fixture"}')
            (source/'layout.npz').write_bytes(b'layout')
            (source/'part.npy').write_bytes(b'shard')
            manifest=dict(layout_sha256=runner.digest(source/'layout.npz'),parts=[dict(path='part.npy',sha256=runner.digest(source/'part.npy'))])
            runner.stage_native_graph(root,work,root/'graph',manifest)
            self.assertEqual((root/'graph/part.npy').read_bytes(),b'shard')
            self.assertEqual((root/'graph/manifest.json').read_text(),(work/'graph_manifest.json').read_text())
            (source/'part.npy').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                runner.stage_native_graph(root,work,root/'changed_graph',manifest)

    def test_cli_requires_all_prerequisites_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            result=subprocess.run([sys.executable,str(Path(runner.__file__)), '--repo',str(root),'--work',str(root/'work'),
                                   '--output',str(root/'out'),'--pulse-search',str(root/'search')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--pulse-search requires',result.stderr)
            self.assertFalse((root/'work').exists())

    def test_coupled_dispatch_forwards_selected_producer(self):
        with mock.patch.object(runner, "replay_coupled", return_value=0) as replay:
            selected = Path("accepted_producer")
            result = runner.check(Path("repo"), Path("work"), Path("output"), Path("build"),
                                  coupled_replay=True, prepare_only=True, producer_software=selected)
        self.assertEqual(result, 0)
        replay.assert_called_once_with(Path("repo"), Path("work"), Path("output"),
                                       Path("build"), True, None, selected)

    def test_cli_forwards_selected_producer_to_native_candidates(self):
        selected = Path("accepted_producer")
        argv = ["runner", "--repo", "repo", "--work", "work", "--output", "out",
                "--native-build", "build", "--canary", "canary", "--pulse-search", "search",
                "--coupled-qualification", "coupled", "--pulse-software", "pulse",
                "--producer-software", str(selected)]
        with mock.patch.object(runner.sys, "argv", argv), \
             mock.patch.object(runner, "execute_pulse_search", return_value=0) as execute:
            with self.assertRaises(SystemExit) as result:
                runner.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(execute.call_args.args[-1], selected)

    def test_saved_search_staging_binds_all_seeds_and_rejects_changed_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';search=root/'search'
            (work/'reference').mkdir(parents=True);(work/'reference_fields').mkdir()
            original=dict(native_library_sha256='b'*64)
            (work/'reference/report.json').write_text(json.dumps(original))
            for path in [work/'reference/initial_parameters.npz',work/'reference/spikes.npz',work/'reference_fields/summary.json',work/'reference_fields/spatial_fields.npz']:
                path.write_bytes(path.name.encode())
            plan=dict(original_report_sha256=runner.digest(work/'reference/report.json'),graph_manifest_sha256='a'*64)
            binding=dict(episode_report_sha256=plan['original_report_sha256'],
                         initial_parameters_sha256=runner.digest(work/'reference/initial_parameters.npz'),spikes_sha256=runner.digest(work/'reference/spikes.npz'),
                         field_summary_sha256=runner.digest(work/'reference_fields/summary.json'),spatial_fields_sha256=runner.digest(work/'reference_fields/spatial_fields.npz'),
                         native_source_identity=dict(native_library_sha256='b'*64,graph_manifest_sha256='a'*64))
            rows=[];seeds=[]
            def save(relative,payload):
                path=search/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(payload)
                rows.append(dict(path=relative,bytes=len(payload),sha256=runner.digest(path)))
            for seed in [41,42,43]:
                name=f'results/seed_{seed}/analysis.json'
                save(name,json.dumps(dict(native_binding=binding,training_seed=seed)).encode())
                seeds.append(dict(training_seed=seed,analysis_sha256=runner.digest(search/name)))
                for name in ['reference.npz','omit.npz','first_future_pulse.npz','original_future_pulses.npz']:
                    save(f'results/seed_{seed}/{name}',b'saved payload')
            save('results/analysis.json',json.dumps(dict(status='THREE_SEED_DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED',completed_seeds=[41,42,43],candidate_results=9,seeds=seeds)).encode())
            save('replay.json',json.dumps(dict(status='INDEPENDENT_THREE_SEED_SAVED_ARRAY_PULSE_SEARCH_REPLAYED',completed_seeds=[41,42,43])).encode())
            proof=dict(status='PASS',exit_code=0,training_seeds=[41,42,43],saved_array_replay_completed=True,training_performed=False,mechanistic_simulation_performed=False,files=rows)
            (search/'delivery.json').write_text(json.dumps(proof))
            reports=runner.stage_pulse_search(search,work,plan)
            self.assertEqual(set(reports),{41,42,43})
            (search/'results/seed_43/omit.npz').write_bytes(b'changed payload')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                runner.stage_pulse_search(search,work,plan)
            proof['training_seeds']=[41,42]
            (search/'delivery.json').write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'completed three-seed'):
                runner.stage_pulse_search(search,work,plan)


if __name__=='__main__': unittest.main()
