"""Exercise admitted CPU04 downstream paths without weakening evidence gates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_population_boundary_closure as boundary
import run_population_forecast_analysis as forecast


class AnalysisPlacementTests(unittest.TestCase):
    def test_CPU04_forecast_retains_three_complete_seed_gate_and_artifact_path(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim') as temp:
            root=Path(temp);work=root/'work';work.mkdir();campaign=root/'campaign'
            for seed in [41,42,43]:
                p=campaign/f'seed_{seed}/delivery.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(status='PASS')))
            args=argparse.Namespace(repo=root,work=work,software=root/'software',campaign=campaign,baseline_analysis=[],prepare_only=False)
            probe=lambda *a,**k:dict(status='PASS');commands=[]
            with patch.dict(os.environ,dict(SLURM_JOB_ID='forecast',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True),patch.object(forecast,'stage_analysis',return_value=probe),patch.object(forecast,'stage_inputs'),patch.object(forecast,'phase',side_effect=lambda w,n,c,p:commands.append(c) or 0):
                self.assertEqual(forecast.run(args),0)
            self.assertEqual(len(commands),1);self.assertIn('--population-campaign',commands[0]);self.assertIn(str(campaign),commands[0])
            report=json.loads((root/'Output/neural_population_dynamics/model/forecast_analysis/run_forecast/delivery.json').read_text())
            self.assertEqual(report['training_seeds'],[41,42,43]);self.assertIs(report['scientific_claim_qualified'],False)
            work=root/'missingwork';work.mkdir();args.work=work;(campaign/'seed_43/delivery.json').unlink()
            with patch.dict(os.environ,dict(SLURM_JOB_ID='missing',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True),patch.object(forecast,'stage_analysis',return_value=probe),patch.object(forecast,'stage_inputs') as stage,patch.object(forecast,'phase') as phase:
                self.assertNotEqual(forecast.run(args),0);stage.assert_not_called();phase.assert_not_called()

    def test_forecast_stages_manifest_bound_costs_without_model_execution(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim') as temp:
            root=Path(temp);software=root/'software';baseline=root/'baseline'
            software.mkdir();baseline.mkdir()
            config=software/'Libs/configs/fixture.yaml';config.parent.mkdir(parents=True)
            config.write_text('mode: development\n')
            def row(base,name):
                p=base/name
                return dict(path=name,bytes=p.stat().st_size,sha256=forecast.digest(p))
            (software/'delivery.json').write_text(json.dumps(dict(status='PASS',exit_code=0,
                tests=dict(tests=35,failures=0,errors=0,skipped=0),
                files=[row(software,'Libs/configs/fixture.yaml')])))
            names=('analysis/analysis.json','analysis/forecast_all_seeds.csv','analysis/costs_by_origin.csv')
            for name in names:
                p=baseline/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('{}\n' if name.endswith('.json') else 'seed,case_id\n41,fixture\n')
            (baseline/'delivery.json').write_text(json.dumps(dict(status='ARCHIVED_VERIFIED_DEVELOPMENT',
                training_seeds=[41,42,43],calibration_outcomes_accessed=False,
                test_outcomes_accessed=False,files=[row(baseline,name) for name in names])))
            accepted=root/'accepted';resource=accepted/'Libs/population_dynamics/resource_probe.py'
            resource.parent.mkdir(parents=True);resource.write_text('def probe(*args, **kwargs): return {"status": "PASS"}\n')
            resources={'Libs/population_dynamics/resource_probe.py':row(accepted,'Libs/population_dynamics/resource_probe.py')}
            work=root/'work';work.mkdir()
            with patch.object(forecast,'prerequisite',return_value=(accepted,{},resources)):
                probe=forecast.stage_analysis(root,work,software,[baseline])
                self.assertEqual(probe()['status'],'PASS')
                for name in names:
                    self.assertEqual((work/'baselines/0'/name).read_bytes(),(baseline/name).read_bytes())
                (baseline/'analysis/costs_by_origin.csv').write_text('changed\n')
                failed=root/'failed';failed.mkdir()
                with self.assertRaisesRegex(ValueError,'source manifest identity changed'):
                    forecast.stage_analysis(root,failed,software,[baseline])
            self.assertFalse((work/'training').exists());self.assertFalse((work/'forecasts').exists())

    def test_CPU04_boundary_retains_inference_only_configuration_and_checkpoint_gate(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim') as temp:
            root=Path(temp);work=root/'work';work.mkdir();args=argparse.Namespace(repo=root,work=work,software=root/'software',campaign=root/'campaign',history=root/'history',prepare_only=False)
            probe=lambda *a,**k:dict(status='PASS');commands=[]
            with patch.dict(os.environ,dict(SLURM_JOB_ID='boundary',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True),patch.object(boundary,'stage_software',return_value=probe),patch.object(boundary,'stage_history'),patch.object(boundary,'stage_inputs'),patch.object(boundary,'stage_checkpoints') as checkpoints,patch.object(boundary,'phase',side_effect=lambda w,n,c,p:commands.append(c) or 0):
                self.assertEqual(boundary.run(args),0);checkpoints.assert_called_once_with(args.campaign,args.work)
            self.assertEqual(len(commands),1);self.assertIn('--campaign',commands[0])
            report=json.loads((root/'Output/neural_population_dynamics/model/boundary_closure/run_boundary/delivery.json').read_text())
            self.assertEqual(report['mode'],'inference_only');self.assertFalse(report['state_selection_enabled']);self.assertFalse(report['formal_tolerances_qualified'])
            self.assertFalse(report['calibration_or_test_accessed'])

    def test_unsupported_CPU_node_GPU_or_prepare_placement_rejected_before_staging(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim') as temp:
            root=Path(temp);args=argparse.Namespace(repo=root,work=root/'work',software=root/'software',campaign=root/'campaign',history=root/'history',baseline_analysis=[],prepare_only=False)
            for module in [forecast,boundary]:
                for node,partition,gpu,prepare in [('mgmt2','cpu','',False),('cpu04','gpu','',False),('cpu04','cpu','0',False),('cpu03','gpu','',True)]:
                    args.prepare_only=prepare
                    with self.subTest(module=module.__name__,node=node),patch.dict(os.environ,dict(SLURM_JOB_ID='bad',SLURMD_NODENAME=node,SLURM_JOB_PARTITION=partition,SLURM_JOB_GPUS=gpu),clear=True),self.assertRaises(ValueError):
                        module.run(args)
            self.assertFalse((root/'work').exists())


class PopulationAnalysisComparisonCallerTests(unittest.TestCase):
    """Reuse completed population aggregates without reopening native inputs."""

    names = ("analysis/analysis.json", "analysis/forecast_all_seeds.csv",
             "analysis/costs_by_origin.csv")

    def test_one_saved_population_with_baselines_uses_comparison_only(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim') as temp:
            root=Path(temp);work=root/'work';work.mkdir()
            args=argparse.Namespace(repo=root,work=work,software=root/'software',campaign=None,
                population_analysis=[root/'population'],baseline_analysis=[root/'baseline20',root/'baseline80'],prepare_only=False)
            commands=[]
            with patch.dict(os.environ,dict(SLURM_JOB_ID='saved',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True),patch.object(forecast,'stage_analysis',return_value=lambda *a,**k:dict(status='PASS')),patch.object(forecast,'stage_population_analyses') as stage,patch.object(forecast,'stage_inputs') as native,patch.object(forecast,'phase',side_effect=lambda w,n,c,p:commands.append(c) or 0):
                self.assertEqual(forecast.run(args),0)
                stage.assert_called_once_with(work,args.population_analysis)
                native.assert_not_called()
            self.assertEqual(commands[0].count('--population-analysis'),1)
            self.assertEqual(commands[0].count('--baseline-analysis'),2)
            self.assertNotIn('--population-campaign',commands[0])

    def make_analysis(self, root, **overrides):
        for name in self.names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n' if name.endswith('.json') else 'seed,case_id\n41,fixture\n')
        receipt = dict(status="PASS", exit_code=0, mode="complete", training_seeds=[41, 42, 43],
                       scientific_claim_qualified=False, calibration_or_test_accessed=False,
                       files=[dict(path=name, bytes=(root / name).stat().st_size,
                                   sha256=forecast.digest(root / name)) for name in self.names])
        receipt.update(overrides)
        (root / "delivery.json").write_text(json.dumps(receipt))
        return root

    def test_comparison_stages_exact_manifest_bound_aggregates_and_rejects_changed_costs(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim") as temp:
            root = Path(temp)
            analyses = [self.make_analysis(root / name) for name in ("fixed", "recurrent")]
            # These files must stay at their original delivered location.
            for analysis in analyses:
                (analysis / "unrequested_checkpoint.bin").write_text("do not stage")
            work = root / "work"
            work.mkdir()
            forecast.stage_population_analyses(work, analyses)
            for i, analysis in enumerate(analyses):
                staged = work / "comparisons" / str(i)
                self.assertEqual(sorted(str(path.relative_to(staged)) for path in staged.rglob("*") if path.is_file()),
                                 sorted((*self.names, "delivery.json")))
                for name in (*self.names, "delivery.json"):
                    self.assertEqual((staged / name).read_bytes(), (analysis / name).read_bytes())
            (analyses[1] / "analysis/costs_by_origin.csv").write_text("changed\n")
            failed = root / "failed"
            failed.mkdir()
            with self.assertRaisesRegex(ValueError, "source manifest identity changed"):
                forecast.stage_population_analyses(failed, analyses)
            self.assertFalse((work / "inputs").exists())
            self.assertFalse((work / "training").exists())
            self.assertFalse((work / "forecasts").exists())

    def test_comparison_rejects_incomplete_or_authoritative_wrappers(self):
        invalid = [dict(status="FAILED"), dict(exit_code=2), dict(mode="prepare"),
                   dict(training_seeds=[41, 42]), dict(scientific_claim_qualified=True),
                   dict(calibration_or_test_accessed=True)]
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim") as temp:
            root = Path(temp)
            good = self.make_analysis(root / "good")
            for i, changes in enumerate(invalid):
                with self.subTest(changes=changes):
                    bad = self.make_analysis(root / ("bad" + str(i)), **changes)
                    work = root / ("work" + str(i))
                    work.mkdir()
                    with self.assertRaisesRegex(ValueError, "complete development-only population analysis required"):
                        forecast.stage_population_analyses(work, [bad, good])
                    self.assertFalse((work / "comparisons").exists())
            single = root/'single';single.mkdir()
            forecast.stage_population_analyses(single, [good])
            self.assertTrue((single/'comparisons/0/delivery.json').exists())
            for count in (0, 3):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "one or two"):
                    forecast.stage_population_analyses(root / "unused", [good] * count)

    def test_comparison_run_forwards_only_aggregates_and_never_stages_native_inputs(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim") as temp:
            root = Path(temp)
            analyses = [self.make_analysis(root / name) for name in ("fixed", "recurrent")]
            work = root / "work"
            work.mkdir()
            args = argparse.Namespace(repo=root, work=work, software=root / "software", campaign=None,
                                      baseline_analysis=[], prepare_only=False, population_analysis=analyses)
            commands = []
            probe = lambda *args, **kwargs: dict(status="PASS")
            environment = dict(SLURM_JOB_ID="comparison", SLURMD_NODENAME="cpu04",
                               SLURM_JOB_PARTITION="cpu", SLURM_JOB_GPUS="")
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(forecast, "stage_analysis", return_value=probe) as software, \
                    patch.object(forecast, "stage_inputs", side_effect=AssertionError("native input staging")) as native, \
                    patch.object(forecast, "phase", side_effect=lambda w, n, c, p: commands.append(c) or 0):
                self.assertEqual(forecast.run(args), 0)
            software.assert_called_once_with(root, work, args.software, [])
            native.assert_not_called()
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][1:], ["-m", "population_dynamics.population_forecast_analysis",
                                             "--population-analysis", "comparisons/0",
                                             "--population-analysis", "comparisons/1", "--output", "analysis"])
            report = json.loads((root / "Output/neural_population_dynamics/model/forecast_analysis/run_comparison/delivery.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["mode"], "comparison")
            self.assertIsNone(report["campaign"])
            self.assertEqual(report["population_analysis"], [str(path) for path in analyses])
            self.assertFalse(report["scientific_claim_qualified"])
            self.assertFalse(report["calibration_or_test_accessed"])
            self.assertFalse((work / "inputs").exists())

    def test_cli_forwards_one_population_analysis_and_two_baselines(self):
        arguments = ["analysis", "--repo", "/tmp/neuronsim/caller/repo", "--work", "/tmp/neuronsim/caller/work",
                     "--software", "/tmp/neuronsim/caller/software",
                     "--population-analysis", "/tmp/neuronsim/caller/population",
                     "--baseline-analysis", "/tmp/neuronsim/caller/baseline20",
                     "--baseline-analysis", "/tmp/neuronsim/caller/baseline80"]
        with patch.object(sys, "argv", arguments), patch.object(forecast, "run", return_value=0) as run:
            with self.assertRaises(SystemExit) as status:
                forecast.main()
        self.assertEqual(status.exception.code, 0)
        run.assert_called_once()
        forwarded = run.call_args.args[0]
        self.assertEqual(forwarded.population_analysis, [Path("/tmp/neuronsim/caller/population")])
        self.assertEqual(forwarded.baseline_analysis, [Path("/tmp/neuronsim/caller/baseline20"),
                                                    Path("/tmp/neuronsim/caller/baseline80")])
        self.assertIsNone(forwarded.campaign)
        self.assertFalse(forwarded.prepare_only)

    def test_cli_forwards_two_analyses_and_rejects_mode_mixing_before_run(self):
        common = ["analysis", "--repo", "/tmp/neuronsim/caller/repo", "--work", "/tmp/neuronsim/caller/work",
                  "--software", "/tmp/neuronsim/caller/software"]
        comparison = ["--population-analysis", "/tmp/neuronsim/caller/fixed",
                      "--population-analysis", "/tmp/neuronsim/caller/recurrent"]
        with patch.object(sys, "argv", common + comparison), patch.object(forecast, "run", return_value=0) as run:
            with self.assertRaises(SystemExit) as status:
                forecast.main()
        self.assertEqual(status.exception.code, 0)
        run.assert_called_once()
        forwarded = run.call_args.args[0]
        self.assertEqual(forwarded.population_analysis, [Path("/tmp/neuronsim/caller/fixed"), Path("/tmp/neuronsim/caller/recurrent")])
        self.assertIsNone(forwarded.campaign)
        invalid = [comparison[:2], comparison + comparison[:2],
                   comparison + ["--campaign", "/tmp/neuronsim/caller/campaign"],
                   comparison + ["--baseline-analysis", "/tmp/neuronsim/caller/baseline"],
                   comparison + ["--prepare-only"]]
        for arguments in invalid:
            with self.subTest(arguments=arguments), patch.object(sys, "argv", common + arguments), patch.object(forecast, "run") as run:
                with self.assertRaises(SystemExit) as status:
                    forecast.main()
                self.assertEqual(status.exception.code, 2)
                run.assert_not_called()

    def test_budget_and_prior_cli_are_explicit_and_reject_incomplete_or_mixed_pairs(self):
        common = ["analysis", "--repo", "/tmp/neuronsim/caller/repo", "--work", "/tmp/neuronsim/caller/work",
                  "--software", "/tmp/neuronsim/caller/software"]
        pair = ["--population-analysis", "/tmp/neuronsim/caller/parent",
                "--population-analysis", "/tmp/neuronsim/caller/child"]
        for flag, attribute, other in [
                ("--budget-comparison", "budget_comparison", "--prior-objective-comparison"),
                ("--prior-objective-comparison", "prior_objective_comparison", "--budget-comparison")]:
            with self.subTest(flag=flag), patch.object(sys, "argv", common + [flag] + pair), \
                    patch.object(forecast, "run", return_value=0) as run:
                with self.assertRaises(SystemExit) as status:
                    forecast.main()
                self.assertEqual(status.exception.code, 0)
                self.assertTrue(getattr(run.call_args.args[0], attribute))
            for arguments in [pair[:2], pair + ["--baseline-analysis", "/tmp/baseline"],
                              ["--campaign", "/tmp/campaign"], pair + ["--prepare-only"], pair + [other]]:
                with self.subTest(flag=flag, arguments=arguments), \
                        patch.object(sys, "argv", common + [flag] + arguments), \
                        patch.object(forecast, "run") as run:
                    with self.assertRaises(SystemExit) as status:
                        forecast.main()
                    self.assertEqual(status.exception.code, 2)
                    run.assert_not_called()

    def test_budget_and_prior_comparisons_forward_saved_aggregates_and_record_mode(self):
        for flag, attribute in [("--budget-comparison", "budget_comparison"),
                                ("--prior-objective-comparison", "prior_objective_comparison")]:
            with self.subTest(flag=flag), tempfile.TemporaryDirectory(dir="/tmp/neuronsim") as temporary:
                root = Path(temporary)
                pair = [self.make_analysis(root / name) for name in ("parent", "child")]
                work = root / "work"
                work.mkdir()
                args = argparse.Namespace(repo=root, work=work, software=root / "software", campaign=None,
                                          baseline_analysis=[], prepare_only=False, population_analysis=pair,
                                          budget_comparison=False, prior_objective_comparison=False)
                setattr(args, attribute, True)
                commands = []
                environment = dict(SLURM_JOB_ID="objective-comparison", SLURMD_NODENAME="cpu04",
                                   SLURM_JOB_PARTITION="cpu", SLURM_JOB_GPUS="")
                with patch.dict(os.environ, environment, clear=True), \
                        patch.object(forecast, "stage_analysis", return_value=lambda *a, **k: dict(status="PASS")), \
                        patch.object(forecast, "stage_inputs", side_effect=AssertionError("native input staging")), \
                        patch.object(forecast, "phase", side_effect=lambda w, n, c, p: commands.append(c) or 0):
                    self.assertEqual(forecast.run(args), 0)
                self.assertEqual(len(commands), 1)
                self.assertIn(flag, commands[0])
                self.assertNotIn("--forecasts", commands[0])
                self.assertFalse((work / "inputs").exists())
                delivery = root / "Output/neural_population_dynamics/model/forecast_analysis/run_objective-comparison/delivery.json"
                record = json.loads(delivery.read_text())
                self.assertIs(record[attribute], True)
                self.assertIs(record["scientific_claim_qualified"], False)

    def test_programmatic_comparison_rejects_mixed_objectives_before_staging(self):
        args = argparse.Namespace(population_analysis=[Path("parent"), Path("child")], campaign=None,
                                  budget_comparison=True, prior_objective_comparison=True)
        with patch.object(forecast, "stage_analysis") as stage, self.assertRaisesRegex(ValueError, "choose"):
            forecast.run(args)
        stage.assert_not_called()


if __name__=='__main__':unittest.main()
