"""Preserve original campaign identities when an unstarted seed moves nodes."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_population_trajectory_campaign as runner


class PopulationTrainingPlacementTests(unittest.TestCase):
    def test_cuda_seed43_keeps_campaign_and_complete_phase_contract(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim',prefix='neuronsim-cuda-placement-') as temp:
            root=Path(temp);repo=root/'repo';work=root/'work';work.mkdir();(work/'Libs/configs').mkdir(parents=True)
            (work/'Libs/configs'/runner.CONFIGS[0]).write_text('training_seeds: [41, 42, 43]\nmaximum_epochs: 20\nearly_stopping_patience: 4\n')
            (work/'Libs/configs'/runner.CONFIGS[1]).write_text('joint_samples: 16\norigins_ms: [1000.0, 3000.0, 5000.0]\nhorizons_ms: [20.0, 80.0, 500.0]\n')
            calls=[]
            def phase(local,name,command,probe):
                calls.append((name,command))
                if name=='fit':runner.write_json(local/'training/run.json',dict(status='FITTED_DEVELOPMENT',training_seed=43,fitting_episodes=56,validation_episodes=16,completed_epochs=20,stop_reason='epoch_budget'))
                elif name=='evaluate':runner.write_json(local/'forecasts/evaluation.json',dict(status='EVALUATED_DEVELOPMENT',score_rows=144))
                else:runner.write_json(local/'replay.json',dict(status='PASS',verified_origins=48))
                return 0
            def probe(*args,**kwargs):return dict(status='PASS')
            environment=dict(SLURM_JOB_ID='cuda-replacement',SLURMD_NODENAME='gpu03',SLURM_JOB_PARTITION='gpu1',SLURM_JOB_GPUS='0')
            with patch.dict(os.environ,environment,clear=True),patch.object(runner,'stage_software',return_value=probe),patch.object(runner,'stage_inputs'),patch.object(runner,'phase',side_effect=phase):
                self.assertEqual(runner.run(repo,work,'cuda',43,'606139','606139_2'),0)
            target=repo/'Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_43'
            delivery=json.loads((target/'delivery.json').read_text());config=json.loads((work/'training_configuration.json').read_text())
            self.assertEqual((delivery['device'],delivery['training_seed'],delivery['array_job_id'],delivery['replaces_unstarted_job']),('cuda',43,'606139','606139_2'))
            self.assertEqual((config['maximum_epochs'],config['early_stopping_patience'],config['device']),(20,4,'cuda'))
            self.assertEqual([name for name,_ in calls],['fit','evaluate','replay'])
            for _,command in calls[1:]:self.assertEqual(command[command.index('--device')+1],'cuda')
            self.assertIn('--all-origins',calls[2][1]);self.assertEqual(delivery['verified_forecast_origins'],48)
            self.assertFalse(delivery['calibration_or_test_accessed']);self.assertFalse(delivery['scientific_claim_qualified'])

    def recovery_fixture(self, root):
        parent, work = root / "parent", root / "work"
        parent.mkdir(); work.mkdir()
        config = dict(training_seeds=[41,42,43], maximum_epochs=20, device="cpu")
        runner.write_json(parent / "training_configuration.json", config)
        runner.write_json(work / "training_configuration.json", config)
        runner.write_json(parent / "training/run.json", dict(
            training_seed=42, completed_epochs=16, effective_maximum_epochs=20,
            calibration_outcomes_accessed=False, test_outcomes_accessed=False))
        for name in ("training/latest.pt", "training/best.pt", "training/epochs.json",
                     "Libs/population_dynamics/population_state.py"):
            path=parent/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"bound-state")
        source=work/"Libs/population_dynamics/population_state.py"
        source.parent.mkdir(parents=True);source.write_bytes(b"bound-state")
        rows=[dict(path=str(path.relative_to(parent)), bytes=path.stat().st_size,
                   sha256=runner.digest(path)) for path in parent.rglob("*") if path.is_file()]
        runner.write_json(parent / "delivery.json", dict(
            status="RESOURCE_BLOCKED", exit_code=3, job_id="607066", array_job_id="606139",
            training_seed=42, device="cpu", representation="recurrent",
            calibration_or_test_accessed=False, scientific_claim_qualified=False, files=rows))
        return parent,work

    def test_resource_recovery_restores_bound_state_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent,work=self.recovery_fixture(Path(tmp))
            before={str(p.relative_to(parent)):runner.digest(p) for p in parent.rglob("*") if p.is_file()}
            info=runner.stage_recovery(parent,work,42,"606139","cpu","recurrent")
            self.assertEqual(info["committed_epochs"],16)
            self.assertEqual(info["maximum_epochs"],20)
            self.assertEqual(info["parent_job_id"],"607066")
            self.assertEqual((work/"training/latest.pt").read_bytes(),b"bound-state")
            self.assertEqual(before,{str(p.relative_to(parent)):runner.digest(p) for p in parent.rglob("*") if p.is_file()})

    def test_recovery_runs_complete_fit_evaluation_and_replay_with_resume(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim",prefix="neuronsim-recovery-flow-") as tmp:
            root=Path(tmp);parent,work=self.recovery_fixture(root);repo=root/"repo"
            (work/"Libs/configs").mkdir(parents=True)
            (work/"Libs/configs"/runner.CONFIGS[0]).write_text("training_seeds: [41,42,43]\nmaximum_epochs: 20\ndevice: cpu\n")
            (work/"Libs/configs"/runner.CONFIGS[1]).write_text("joint_samples: 16\norigins_ms: [1000.0,3000.0,5000.0]\nhorizons_ms: [20.0,80.0,500.0]\n")
            calls=[]
            def phase(local,name,command,probe):
                calls.append((name,command))
                if name=="fit":runner.write_json(local/"training/run.json",dict(status="FITTED_DEVELOPMENT",training_seed=42,fitting_episodes=56,validation_episodes=16,completed_epochs=20,stop_reason="epoch_budget"))
                elif name=="evaluate":runner.write_json(local/"forecasts/evaluation.json",dict(status="EVALUATED_DEVELOPMENT",score_rows=144))
                else:runner.write_json(local/"replay.json",dict(status="PASS",verified_origins=48))
                return 0
            env=dict(SLURM_JOB_ID="recovery",SLURMD_NODENAME="cpu04",SLURM_JOB_PARTITION="cpu",SLURM_JOB_GPUS="")
            info=dict(parent_job_id="607066",committed_epochs=16,maximum_epochs=20)
            with patch.dict(os.environ,env,clear=True),patch.object(runner,"stage_software",return_value=lambda *a,**k:dict(status="PASS")),patch.object(runner,"stage_inputs"),patch.object(runner,"stage_recovery",return_value=info) as restore,patch.object(runner,"phase",side_effect=phase):
                self.assertEqual(runner.run(repo,work,"cpu",42,"606139","606139_1",recovery_from=parent),0)
            restore.assert_called_once_with(parent.resolve(),work,42,"606139","cpu","recurrent")
            self.assertEqual([name for name,_ in calls],["fit","evaluate","replay"])
            self.assertIn("--resume",calls[0][1]);self.assertNotIn("--total-epochs",calls[0][1])
            self.assertIn("--all-origins",calls[2][1])
            target=repo/"Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_42"
            proof=json.loads((target/"delivery.json").read_text())
            self.assertEqual(proof["resource_recovery"],info);self.assertEqual(proof["verified_forecast_origins"],48)

    def test_resource_recovery_rejects_seed_device_budget_source_and_state_changes(self):
        for change in ("seed", "device", "budget", "source", "checkpoint", "test_access"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                parent,work=self.recovery_fixture(Path(tmp))
                seed,device=42,"cpu"
                if change=="seed":seed=43
                elif change=="device":device="cuda"
                elif change=="budget":runner.write_json(work/"training_configuration.json",dict(maximum_epochs=21))
                elif change=="source":(work/"Libs/population_dynamics/population_state.py").write_text("changed")
                elif change=="checkpoint":(parent/"training/latest.pt").write_text("corrupt")
                elif change=="test_access":
                    proof=json.loads((parent/"delivery.json").read_text());proof["calibration_or_test_accessed"]=True
                    runner.write_json(parent/"delivery.json",proof)
                with self.assertRaises(ValueError):runner.stage_recovery(parent,work,seed,"606139",device,"recurrent")

    def test_existing_array_identity_and_allowed_CPU_nodes(self):
        for node,partition in [('cpu03','gpu'),('cpu04','cpu')]:
            for task,seed in enumerate([41,42,43]):
                with self.subTest(node=node,seed=seed),patch.dict(os.environ,dict(SLURM_JOB_ID='606140',SLURM_ARRAY_JOB_ID='606139',SLURM_ARRAY_TASK_ID=str(task),SLURMD_NODENAME=node,SLURM_JOB_PARTITION=partition,SLURM_JOB_GPUS=''),clear=True):
                    result=runner.execution_identity('cpu')
                self.assertEqual(result['seed'],seed);self.assertEqual(result['array_id'],'606139')
                self.assertIsNone(result['replaces_job'])

    def test_explicit_relocation_keeps_original_seed_campaign_and_replaced_job(self):
        with patch.dict(os.environ,dict(SLURM_JOB_ID='replacement',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True):
            for seed,task in [(42,1),(43,2)]:
                result=runner.execution_identity('cpu',seed,'606139',f'606139_{task}')
                self.assertEqual(result['seed'],seed);self.assertEqual(result['array_id'],'606139');self.assertEqual(result['task'],task)
                self.assertEqual(result['replaces_job'],f'606139_{task}')
            for args in [(42,None,None),(42,'606139',None),(44,'606139','606139_3'),(42,'606139','606139_2'),(42,'bad','bad_1')]:
                with self.subTest(args=args),self.assertRaises(ValueError):runner.execution_identity('cpu',*args)

    def test_invalid_placement_or_GPU_requests_are_rejected(self):
        for node,partition,gpu,device in [('mgmt2','cpu','','cpu'),('cpu04','gpu','','cpu'),('cpu03','gpu','0','cpu'),('gpu03','gpu1','0,1','cuda'),('gpu03','gpu1','','cuda')]:
            with self.subTest(node=node,gpu=gpu),patch.dict(os.environ,dict(SLURM_JOB_ID='job',SLURMD_NODENAME=node,SLURM_JOB_PARTITION=partition,SLURM_JOB_GPUS=gpu,SLURM_ARRAY_JOB_ID='606139',SLURM_ARRAY_TASK_ID='0'),clear=True),self.assertRaises(ValueError):
                runner.execution_identity(device)

    def test_complete_relocated_runner_uses_same_outputs_inputs_and_all_phases(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim',prefix='neuronsim-placement-') as temp:
            root=Path(temp);repo=root/'repo';work=root/'work';work.mkdir();(work/'Libs/configs').mkdir(parents=True)
            (work/'Libs/configs'/runner.CONFIGS[0]).write_text('training_seeds: [41, 42, 43]\n')
            (work/'Libs/configs'/runner.CONFIGS[1]).write_text('joint_samples: 16\norigins_ms: [1000.0, 3000.0, 5000.0]\nhorizons_ms: [20.0, 80.0, 500.0]\n')
            calls=[]
            def phase(local,name,command,probe):
                calls.append(name)
                if name=='fit':runner.write_json(local/'training/run.json',dict(status='FITTED_DEVELOPMENT',training_seed=42,fitting_episodes=56,validation_episodes=16,completed_epochs=20,stop_reason='maximum_epochs'))
                elif name=='evaluate':runner.write_json(local/'forecasts/evaluation.json',dict(status='EVALUATED_DEVELOPMENT',score_rows=144))
                else:runner.write_json(local/'replay.json',dict(status='PASS',verified_origins=48))
                return 0
            def probe(*args,**kwargs):return dict(status='PASS')
            with patch.dict(os.environ,dict(SLURM_JOB_ID='replacement',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True),patch.object(runner,'stage_software',return_value=probe),patch.object(runner,'stage_inputs'),patch.object(runner,'phase',side_effect=phase):
                code=runner.run(repo,work,'cpu',42,'606139','606139_1')
            target=repo/'Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_42'
            delivery=json.loads((target/'delivery.json').read_text())
            self.assertEqual(code,0);self.assertEqual(calls,['fit','evaluate','replay']);self.assertEqual(delivery['training_seed'],42)
            self.assertEqual(delivery['array_job_id'],'606139');self.assertEqual(delivery['replaces_unstarted_job'],'606139_1')
            self.assertEqual(delivery['fit_episodes'],56);self.assertEqual(delivery['validation_episodes'],16)
            self.assertEqual(delivery['verified_forecast_origins'],48);self.assertEqual(delivery['episode_ms'],6000)
            with patch.dict(os.environ,dict(SLURM_JOB_ID='other',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS=''),clear=True),self.assertRaises(FileExistsError):
                runner.run(repo,work,'cpu',42,'606139','606139_1')

    def test_invalid_relocation_real_CLI_exits_before_writes(self):
        with tempfile.TemporaryDirectory(dir='/tmp/neuronsim',prefix='neuronsim-placement-cli-') as temp:
            root=Path(temp)
            environment=dict(os.environ,SLURM_JOB_ID='replacement',SLURMD_NODENAME='cpu04',SLURM_JOB_PARTITION='cpu',SLURM_JOB_GPUS='')
            environment.pop('SLURM_ARRAY_JOB_ID',None)
            result=subprocess.run([sys.executable,str(Path(runner.__file__)),'--repo',str(root),'--work',str(root/'work'),'--seed','42','--campaign-array-id','606139','--replaces-job','606139_2'],env=environment,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('replaced job must match',result.stderr);self.assertFalse((root/'work').exists())


if __name__=='__main__':unittest.main()
