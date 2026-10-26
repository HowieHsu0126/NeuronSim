"""Native adapter fixtures exercise metadata and publication, never scientific simulation."""
from __future__ import annotations

from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from . import pfc_native_reference_execution as adapter
from . import pfc_source_comparison as comparison
from . import pfc_scale_analysis as source
from . import test_pfc_source_comparison as comparison_fixtures
from .provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]


class NativeReferenceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = comparison_fixtures.ReferenceClockCorrespondenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        pinned = self.root/'pinned_source'
        for name in adapter.SOURCE_FILES:
            target = pinned/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'Libs/external/pfc_working_memory'/name,target)
        self.native_python = self.root/'native python'
        self.native_python.write_text('#!/bin/sh\nexit 1\n');self.native_python.chmod(0o755)
        self.inputs = adapter.ExecutionInputs(self.fixture.prepared,self.fixture.config_path,self.fixture.sources,
            self.fixture.proof,pinned,self.native_python,self.root/'native_work',self.root/'native_delivery',
            self.root/'matplotlib',ROOT)
        self.addCleanup(shutil.rmtree,self.inputs.work,ignore_errors=True)
        self.calls=[]
        self.native_returncode=0
        self.output_mode='complete'
        self.runtime=dict(python='3.10.14 engineering fixture',interpreter=str(self.native_python),
            brian2='2.5.4',numpy='1.26.4',scipy='1.12.0')
        self.environment=dict(SLURM_JOB_ID='engineering_fixture',SLURMD_NODENAME='cpu03',
            SLURM_JOB_PARTITION='gpu',SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='1',SLURM_MEM_PER_NODE='8192')

    def fake_native(self,command,*,cwd,env,**kwargs):
        self.calls.append(command)
        work=Path(cwd)
        self.assertEqual(command[0],str(self.inputs.native_python))
        self.assertEqual(env['MPLCONFIGDIR'],str(self.inputs.native_matplotlib_cache))
        self.assertEqual(env['OMP_NUM_THREADS'],'1')
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')
        self.assertTrue(all((work/'Libs/population_dynamics'/(name+'.py')).is_file() for name in adapter.NATIVE_MODULES))
        self.assertTrue((work/'Libs/population_dynamics/pfc_kernel_parameters.py').exists())
        self.assertTrue(all((work/'source'/name).is_file() for name in adapter.SOURCE_FILES))
        if command[-1]=='--preflight-help':
            runtime=dict(self.runtime,native_cli_import_help=True,source_entry_sha256=source.SOURCE_ENTRY_SHA256)
            return subprocess.CompletedProcess(command,0,stdout='help fixture\n'+adapter.RUNTIME_PREFIX+json.dumps(runtime)+'\n',stderr='')
        self.assertEqual(command[3:5],['--source',str(work/'source')])
        kwargs['stdout'].write(adapter.RUNTIME_PREFIX+json.dumps(self.runtime)+'\n')
        native=work/'native';native.mkdir()
        if self.native_returncode:
            (native/'partial.txt').write_text('retained failed native fixture')
            return subprocess.CompletedProcess(command,self.native_returncode)
        if self.output_mode=='missing_report':
            return subprocess.CompletedProcess(command,0)
        config=yaml.safe_load((work/'native_config.yaml').read_text())
        cases=[]
        for item in config['cases']:
            raw=native/(item['case_id']+'_native.npz')
            raw.write_bytes(b'non-scientific retained raw fixture')
            cases.append(dict(item,core_units=100000,native_events=2,elapsed_seconds=1.,
                raw_file=raw.name,raw_sha256='0'*64 if self.output_mode=='bad_raw_binding' else sha256_file(raw)))
        if self.output_mode=='missing_case':cases.pop()
        if self.output_mode=='wrong_unit':cases[0]['core_units']=99999
        report=dict(status='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',cases=cases,resolved_config=config,
            source_revision=source.SOURCE_REVISION,source_entry_sha256=source.SOURCE_ENTRY_SHA256,
            synapse_execution='factorized_uniform_projections',
            execution_implementation_sha256=self.fixture.design['reference_execution_implementation_sha256'],
            full_scale_qualified=False,scientific_fidelity_qualified=False,brian2_version='2.5.4',numpy_version='1.26.4')
        if self.output_mode=='wrong_report_runtime':report['numpy_version']='2.4.2'
        if self.output_mode=='wrong_config':report['resolved_config']=dict(config,dt_ms=.05)
        self.fixture.write(native/'report.json',report)
        (native/'dataset').mkdir()
        self.fixture.write(native/'dataset/dataset.json',dict(complete_episodes=len(cases)))
        return subprocess.CompletedProcess(command,0)

    def run_adapter(self,*,preflight_only=False):
        with patch.dict(os.environ,self.environment,clear=True),patch.object(adapter.subprocess,'run',side_effect=self.fake_native):
            return adapter.run(self.inputs,preflight_only=preflight_only)

    def arguments(self,mode='--preflight-only'):
        inputs=self.inputs
        args=['--preparation',str(inputs.preparation),'--canonical-config',str(inputs.canonical_config),
            '--factorization-proof',str(inputs.factorization_proof),'--pinned-source',str(inputs.pinned_source),
            '--native-python',str(inputs.native_python),'--work',str(inputs.work),'--output',str(inputs.output),
            '--native-matplotlib-cache',str(inputs.native_matplotlib_cache),'--project-root',str(ROOT),mode]
        for path in inputs.source_runs:args+=['--source-run',str(path)]
        return args

    def test_preflight_real_entry_stages_help_only_and_creates_no_requested_outputs(self):
        with patch.object(adapter.subprocess,'run',side_effect=self.fake_native),patch('sys.stdout',new=io.StringIO()) as out:
            self.assertEqual(adapter.main(self.arguments()),0)
        report=json.loads(out.getvalue())
        self.assertEqual(report['status'],'PREFLIGHT_PASS_NATIVE_REFERENCE_CLOCK')
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][-1],'--preflight-help')
        self.assertFalse(Path(self.calls[0][2]).parent.exists())
        self.assertFalse(self.inputs.work.exists());self.assertFalse(self.inputs.output.exists())
        self.assertFalse(report['native_execution_performed']);self.assertFalse(report['source_archive_created'])
        self.assertIn('pfc_kernel_parameters',report['staged_modules'])

    def test_absolute_venv_symlink_invocation_is_not_resolved(self):
        alias=self.root/'venv/bin/python';alias.parent.mkdir(parents=True);alias.symlink_to(self.native_python)
        self.inputs=replace(self.inputs,native_python=alias)
        self.runtime['interpreter']=str(alias)
        self.run_adapter(preflight_only=True)
        self.assertEqual(self.calls[0][0],str(alias))
        self.assertNotEqual(self.calls[0][0],str(alias.resolve()))

    def test_complete_execution_delivery_satisfies_current_consumer_metadata(self):
        record=self.run_adapter()
        self.assertEqual(record['status'],'NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE')
        self.assertEqual(record['exit_code'],0);self.assertEqual(record['native_exit_code'],0)
        self.assertFalse(self.inputs.work.exists())
        output=self.inputs.output
        delivery=source.read(output/'delivery.json')
        index={row['path']:row for row in delivery['files']}
        for name in ('native/report.json','native_config.yaml','execution.json','Libs/population_dynamics/pfc_factorized_source.py'):
            self.assertIn(name,index)
            self.assertEqual(index[name]['sha256'],sha256_file(output/name))
        execution=source.read(output/'execution.json')
        self.assertEqual(execution['native_report_sha256'],sha256_file(output/'native/report.json'))
        self.assertEqual(execution['design_sha256'],sha256_file(self.inputs.preparation/'design.json'))
        self.assertEqual(execution['native_config_sha256'],self.fixture.design['native_config_sha256'])
        self.assertEqual(execution['runtime_versions'],self.runtime)
        self.assertTrue(all(record[key]==value for key,value in adapter.QUALIFICATION_FLAGS.items()))
        with patch.object(source,'_source_values',side_effect=RuntimeError('STOP_BEFORE_SCIENTIFIC_ARRAYS')):
            with self.assertRaisesRegex(RuntimeError,'STOP_BEFORE_SCIENTIFIC_ARRAYS'):
                comparison.compare(None,[output/'native'],self.inputs.preparation/'design.json',self.root/'comparison.json',
                    candidate_kind='streamed_numpy',config_path=self.inputs.canonical_config,
                    source_runs=self.inputs.source_runs,factorization_proof=self.inputs.factorization_proof)

    def test_nonzero_native_exit_is_retained_and_published_without_retry(self):
        self.native_returncode=7
        record=self.run_adapter()
        self.assertEqual(record['exit_code'],7);self.assertEqual(record['native_exit_code'],7)
        self.assertEqual(record['status'],'NATIVE_REFERENCE_CLOCK_EXECUTION_FAILED')
        self.assertEqual(len(self.calls),2)
        self.assertTrue((self.inputs.work/'native/partial.txt').exists())
        self.assertTrue((self.inputs.output/'native/partial.txt').exists())
        self.assertEqual(source.read(self.inputs.output/'delivery.json')['exit_code'],7)

    def test_zero_exit_with_incomplete_or_drifted_outputs_is_rejected(self):
        for mode in ('missing_report','missing_case','wrong_unit','wrong_config','wrong_report_runtime'):
            with self.subTest(mode=mode):
                self.output_mode=mode
                self.inputs=replace(self.inputs,work=self.root/(mode+'_work'),output=self.root/(mode+'_output'))
                self.addCleanup(shutil.rmtree,self.inputs.work,ignore_errors=True)
                record=self.run_adapter()
                self.assertEqual(record['native_exit_code'],0)
                self.assertNotEqual(record['exit_code'],0)
                self.assertEqual(record['status'],'NATIVE_REFERENCE_CLOCK_EXECUTION_FAILED')
                self.assertTrue(self.inputs.work.exists())

    def test_archive_failure_retains_actual_native_success_and_nonaccepting_delivery(self):
        def failed_copy(work,output):
            output.mkdir();shutil.copy2(work/'execution.json',output/'execution.json')
            raise OSError('fixture copy failure')
        with patch.object(adapter,'archive_tree',side_effect=failed_copy):
            record=self.run_adapter()
        self.assertEqual(record['native_exit_code'],0);self.assertEqual(record['exit_code'],2)
        self.assertEqual(record['archive_status'],'FAILED')
        self.assertTrue(self.inputs.work.exists())
        self.assertEqual(source.read(self.inputs.output/'delivery.json')['files'],[])

    def test_producer_raw_hash_must_match_existing_verified_copy_index(self):
        self.output_mode='bad_raw_binding'
        record=self.run_adapter()
        self.assertEqual(record['status'],'NATIVE_REFERENCE_ARCHIVE_FAILED')
        self.assertEqual(record['native_exit_code'],0)
        self.assertEqual(source.read(self.inputs.output/'delivery.json')['files'],[])
        self.assertTrue(self.inputs.work.exists())

    def test_preflight_rejects_config_projection_or_source_binding_drift_without_arrays(self):
        config_path=self.inputs.preparation/'native_config.yaml'
        config=yaml.safe_load(config_path.read_text());config['dt_ms']=.05
        config_path.write_text(yaml.safe_dump(config))
        with patch.object(adapter.subprocess,'run') as child:
            with self.assertRaisesRegex(ValueError,'projection changed'):
                adapter.preflight(self.inputs)
            child.assert_not_called()

    def test_preflight_rejects_source_and_factorization_proof_drift(self):
        report_path=self.inputs.source_runs[0]/'report.json'
        report=source.read(report_path);report['driver']['numpy_version']='other'
        self.fixture.write(report_path,report);self.fixture.fixture.rebind(self.inputs.source_runs[0])
        with patch.object(adapter.subprocess,'run') as child:
            with self.assertRaisesRegex(ValueError,'source report/delivery changed'):
                adapter.preflight(self.inputs)
            child.assert_not_called()

    def test_wrong_native_runtime_is_rejected_after_guarded_help_without_simulation(self):
        self.runtime['scipy']='other'
        with self.assertRaisesRegex(ValueError,'runtime differs'):
            self.run_adapter(preflight_only=True)
        self.assertEqual(len(self.calls),1)
        self.assertFalse(self.inputs.work.exists());self.assertFalse(self.inputs.output.exists())

    def test_fresh_disjoint_local_destinations_and_explicit_interpreter_are_required(self):
        for changes in (dict(work=Path('relative')),dict(native_python=Path('python')),
                        dict(work=self.inputs.preparation/'new_work'),dict(output=self.inputs.work/'nested'),
                        dict(work=Path('/share/not_node_local'))):
            with self.subTest(changes=changes),patch.object(adapter.subprocess,'run') as child:
                with self.assertRaises(ValueError):adapter.preflight(replace(self.inputs,**changes))
                child.assert_not_called()
        self.inputs.work.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh native work'):
            adapter.preflight(self.inputs)

    def test_actual_execution_requires_cpu03_no_gpu_one_cpu_and_eight_gib(self):
        for key,value in (('SLURMD_NODENAME','cpu04'),('SLURM_JOB_GPUS','0'),('SLURM_CPUS_PER_TASK','2'),('SLURM_MEM_PER_NODE','4096')):
            with self.subTest(key=key),patch.dict(os.environ,dict(self.environment,**{key:value}),clear=True), \
                 patch.object(adapter.subprocess,'run',side_effect=self.fake_native):
                with self.assertRaises(ValueError):adapter.run(self.inputs,preflight_only=False)
                self.assertFalse(self.inputs.work.exists())

    def test_existing_failed_work_or_output_prevents_unknown_operation_repeat(self):
        self.native_returncode=7;self.run_adapter()
        calls=len(self.calls)
        with self.assertRaisesRegex(ValueError,'fresh native work'):
            self.run_adapter()
        self.assertEqual(len(self.calls),calls)


class NativeReferenceLauncherTests(unittest.TestCase):
    def test_wrapper_defaults_and_explicit_controller_passthrough(self):
        path=ROOT/'Libs/scripts/run_pfc_native_reference.sbatch'
        text=path.read_text()
        for directive in ('--partition=gpu','--nodelist=cpu03','--ntasks=1','--cpus-per-task=1','--mem=8G','--gres=none','--time=0'):
            self.assertIn('#SBATCH '+directive,text)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);capture=root/'capture.json';script=root/'capture.py'
            script.write_text('import json,sys\nfrom pathlib import Path\nPath('+repr(str(capture))+').write_text(json.dumps(sys.argv[1:]))\n')
            interpreter=root/'controller';interpreter.write_text('#!/bin/sh\nexec '+str(sys.executable)+' -B '+str(script)+' "$@"\n');interpreter.chmod(0o755)
            env=dict(PATH=os.environ['PATH'],NEURONSIM_PFC_PYTHON=str(interpreter),NEURONSIM_PROJECT_ROOT=str(ROOT))
            result=subprocess.run(['/bin/bash',str(path),'--preflight-only','--work','/tmp/fresh'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(capture.read_text()),['-B','-m','population_dynamics.pfc_native_reference_execution','--preflight-only','--work','/tmp/fresh'])
            env.pop('NEURONSIM_PFC_PYTHON')
            result=subprocess.run(['/bin/bash',str(path)],env=env,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('controller Python',result.stderr)


if __name__=='__main__':
    unittest.main()
