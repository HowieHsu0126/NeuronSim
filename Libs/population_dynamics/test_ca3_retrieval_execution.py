import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import ca3_retrieval_execution as execution
from .carlsim_records import (GROUP_HEADER, NETWORK_HEADER, NETWORK_SIGNATURE,
                              SOURCE_REVISION, validate_retrieval)
from .provenance import sha256_file
from .test_carlsim_records import events, network
from .test_carlsim_formation_replay import coupled_declaration, coupled_network


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RetrievalInput(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.formation = self.root/'formation'
        (self.formation/'native_run').mkdir(parents=True)
        self.raw = self.formation/'native_run/network65p.dat'
        self.raw.write_bytes(NETWORK_HEADER.pack(NETWORK_SIGNATURE,.3,45.85,1.,84053,1)
                            + GROUP_HEADER.pack(0,84052,84053,1,1,b'CA3_Pyramidal')
                            + struct.pack('<ii',1,1)+struct.pack('<iiiiiffi',0,0,0,1,50,1.,2.,1))
        self.normalized = self.root/execution.INPUT_NAME
        shutil.copy2(self.raw, self.normalized)
        self.native_path = self.formation/'native_report.json'
        write(self.native_path, dict(status='NATIVE_FORMATION_EXIT_ZERO',exit_code=0,
              expected_duration_ms=70000,core_neurons=84053,author_revision=execution.AUTHOR_REVISION,
              framework_revision=SOURCE_REVISION,complete_author_formation_program_unchanged=True,
              complete_formation_executed=True,populations={'CA3_Pyramidal':84053},pattern_sha256='pattern'))
        write(self.formation/'observations/recording.json', {'fixture':True})
        self.form_replay = self.root/'formation_replay.json'
        write(self.form_replay, dict(status='PASS_NATIVE_FORMATION_PREFIX_REPLAY_AND_COUNTS',
              formation_report_sha256=sha256_file(self.native_path),
              formation_recording_sha256=sha256_file(self.formation/'observations/recording.json')))
        self.normal_report = self.root/'normalization.json'
        write(self.normal_report, dict(status='APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION',
              source_sha256=sha256_file(self.raw),output_sha256=sha256_file(self.normalized)))
        self.normal_replay = self.root/'normalization_replay.json'
        write(self.normal_replay, dict(status='PASS_NORMALIZATION_REPLAY',all_record_bytes_verified=True,
              normalization_report_sha256=sha256_file(self.normal_report),source_sha256=sha256_file(self.raw),
              normalized_sha256=sha256_file(self.normalized),weight_tolerance=0,target_global_mean=.625))

    def tearDown(self):
        self.temp.cleanup()

    def qualify(self):
        return execution.qualify_input(self.formation,self.form_replay,self.normalized,
                                       self.normal_report,self.normal_replay)

    def test_receipt_binding_and_changed_source_are_enforced(self):
        self.assertTrue(self.qualify()['normalized_input_qualified'])
        self.raw.write_bytes(self.raw.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'NORMALIZATION_REPLAY_REQUIRED'):
            self.qualify()

    def test_partial_formation_or_failed_normalization_cannot_launch(self):
        for path in (self.form_replay,self.normal_replay):
            with self.subTest(path=path.name):
                original = path.read_text()
                value = json.loads(original);value['status']='FAILED'
                write(path,value)
                with self.assertRaisesRegex(ValueError,'REQUIRED'):
                    self.qualify()
                path.write_text(original)


class RetrievalExecution(unittest.TestCase):
    def test_observation_build_flag_and_fresh_output_are_bound(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root=Path(temp)
            self.assertEqual(execution.observation_environment(root,{},enabled=False),{})
            with self.assertRaisesRegex(ValueError,'FLAGS_DIFFER'):
                execution.observation_environment(root,{},enabled=True)
            report=dict(native_observation_overlay=dict(
                enabled_by=execution.ENVIRONMENT_VARIABLE,framework_revision=SOURCE_REVISION,
                original_sha256=execution.PINNED_FILES,native_build_verified=True),
                observer_implementation_sha256=sha256_file(
                    Path(execution.__file__).with_name('ca3_native_observation.py')))
            with self.assertRaisesRegex(ValueError,'FLAGS_DIFFER'):
                execution.observation_environment(root,report,enabled=False)
            self.assertEqual(execution.observation_environment(root,report,enabled=True),
                             {execution.ENVIRONMENT_VARIABLE:str(root/'native_observation.bin')})
            report['native_observation_overlay']['native_build_verified']=False
            with self.assertRaisesRegex(ValueError,'VERIFIED_OBSERVATION_BUILD'):
                execution.observation_environment(root,report,enabled=True)
            report['native_observation_overlay']['native_build_verified']=True
            (root/'native_observation.bin').write_bytes(b'preserved')
            with self.assertRaises(FileExistsError):
                execution.observation_environment(root,report,enabled=True)
            self.assertEqual((root/'native_observation.bin').read_bytes(),b'preserved')

    def test_ambient_observation_target_cannot_silently_change_execution(self):
        with patch.dict(os.environ,{execution.ENVIRONMENT_VARIABLE:'/existing'}):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_RUNNER_BINDING'):
                execution.observation_environment(Path('/missing'),{},enabled=False)

    def test_exit_failure_is_recorded_without_decoding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'bin').mkdir()
            shutil.copy2('/bin/false',root/'bin/ca3_retrieval')
            (root/'trainPat.csv').write_text('0 1\n')
            binding=dict(normalized_input_qualified=True,normalized_input_sha256='input',
                         populations={'CA3_Pyramidal':84053},pattern_sha256=sha256_file(root/'trainPat.csv'))
            write(root/'build_report.json',dict(status='NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING',
                  scope='native_source_retrieval',expected_duration_ms=1001,author_revision=execution.AUTHOR_REVISION,
                  framework_revision=SOURCE_REVISION,retrieval_program_changed_only_in_input_path=True,
                  binary_sha256=sha256_file(root/'bin/ca3_retrieval'),populations=binding['populations'],
                  pattern_sha256=binding['pattern_sha256'],scientific_fidelity_qualified=False,
                  retrieval_input_contract=dict(effective_input_binding=execution.INPUT_NAME,source_path_binding_only=True,
                  source_carlsim_seed=10,source_host_srand_seed=3,source_estdp_enabled=True)))
            write(root/'manifest.json',dict(files=[dict(path=n,sha256=sha256_file(root/n))
                  for n in ('build_report.json','trainPat.csv','bin/ca3_retrieval')]))
            args=(root,root/'formation',root/'f_replay',root/'input',root/'normal',root/'n_replay')
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}), \
                 patch.object(execution,'qualify_input',return_value=binding), \
                 patch.object(execution,'probe',return_value={'status':'PASS'}), \
                 patch.object(execution,'validate_retrieval') as decoder:
                with self.assertRaisesRegex(RuntimeError,'RETRIEVAL_EXIT:1'):
                    execution.execute(*args)
                decoder.assert_not_called()
                report=json.loads((root/'native_report.json').read_text())
                self.assertEqual(report['exit_code'],1)
                self.assertEqual(report['status'],'FAILED')
                self.assertFalse(report['retrieval_executed'])
                with self.assertRaises(FileExistsError):
                    execution.execute(*args)

    def test_missing_allocation_is_rejected(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(ValueError,'ALLOCATED_GPU'):
                execution.execute(*([Path('/missing')]*6))


class RetrievalRecords(unittest.TestCase):
    def test_input_formation_clock_and_retrieval_event_clock_stay_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'native_run';(run/'results').mkdir(parents=True)
            network(run/execution.INPUT_NAME,seconds=45.85)
            events(run/'results/spk_I.dat',2,[(0,0),(1000,1)])
            events(run/'results/spk_E.dat',3,[(520,2),(520,2),(1,0)])
            report=dict(status='NATIVE_RETRIEVAL_EXIT_ZERO',exit_code=0,framework_revision=SOURCE_REVISION,
                        scope='native_source_retrieval',native_run_executed=True,expected_duration_ms=1001,
                        normalized_input_qualified=True,normalized_input_sha256=sha256_file(run/execution.INPUT_NAME),
                        core_neurons=5,populations={'I':2,'E':3})
            write(root/'native_report.json',report)
            result=validate_retrieval(root,root/'observations')
            self.assertEqual(result['records'],5)
            self.assertTrue(result['retrieval_executed'])
            self.assertFalse(result['terminal_native_clock_independently_verified'])
            with np.load(root/'observations/population_counts.npz') as data:
                self.assertEqual(data['population_counts_per_ms'].shape,(1001,2))
                np.testing.assert_array_equal(data['unit_counts'],[1,1,1,0,2])
            report['normalized_input_qualified']=False
            write(root/'native_report.json',report)
            with self.assertRaisesRegex(ValueError,'QUALIFIED_SOURCE_RETRIEVAL'):
                validate_retrieval(root,root/'rejected')
            self.assertFalse((root/'rejected').exists())


class CoupledRetrievalInput(RetrievalInput):
    """Receipt-binding fixtures, not evidence that normalization was executed."""
    def setUp(self):
        super().setUp()
        self.old_reference_input = self.normalized
        self.normalized = self.root/'network65p_assem550_divnorm.dat'
        coupled_network(self.raw, seconds=45.85)
        shutil.copy2(self.raw, self.normalized)
        self.pattern = self.root/'trainPat.csv'
        self.pattern.write_text(' '.join(str(i) for i in list(range(550))*4)+'\n')
        self.extension = coupled_declaration()
        self.extension['derived_pattern_sha256'] = sha256_file(self.pattern)
        native = json.loads(self.native_path.read_text())
        native.update(status='COUPLED_EXTENSION_FORMATION_EXIT_ZERO', scope='coupled_extension_formation',
            core_neurons=168106, populations=self.extension['derived_populations'],
            complete_author_formation_program_unchanged=False,
            source_code_semantics_preserved_before_stop=False,
            coupled_extension=self.extension,
            coupled_extension_config=dict(resolved={'population_multiplier':2},sha256='8'*64),
            pattern_sha256=sha256_file(self.pattern))
        write(self.native_path,native)
        replay=json.loads(self.form_replay.read_text())
        replay.update(status='PASS_COUPLED_EXTENSION_FORMATION_PREFIX_REPLAY_AND_COUNTS',
                      formation_report_sha256=sha256_file(self.native_path),coupled_extension=self.extension)
        write(self.form_replay,replay)
        self.rebind_normalization()

    def rebind_normalization(self):
        write(self.normal_report,dict(status='APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION',
            source_sha256=sha256_file(self.raw),output_sha256=sha256_file(self.normalized)))
        write(self.normal_replay,dict(status='PASS_NORMALIZATION_REPLAY',all_record_bytes_verified=True,
            normalization_report_sha256=sha256_file(self.normal_report),source_sha256=sha256_file(self.raw),
            normalized_sha256=sha256_file(self.normalized),weight_tolerance=0,target_global_mean=.625))

    def qualify(self, *, coupled_extension=True, source_realization=None):
        return execution.qualify_input(self.formation,self.form_replay,self.normalized,
                                       self.normal_report,self.normal_replay,coupled_extension=coupled_extension,
                                       source_realization=source_realization)

    def runtime(self):
        root=self.root/'runtime';(root/'bin').mkdir(parents=True)
        shutil.copy2('/bin/false',root/'bin/ca3_retrieval')
        shutil.copy2(self.pattern,root/'trainPat.csv')
        extension=dict(self.extension,source_role='retrieval',source_cue_size=138,derived_cue_size=276)
        build=dict(status='COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED',
            scope='coupled_extension_retrieval',expected_duration_ms=1001,
            author_revision=execution.AUTHOR_REVISION,framework_revision=SOURCE_REVISION,
            core_neurons=168106,populations=extension['derived_populations'],
            source_code_semantics_preserved_before_stop=False,coupled_extension=extension,
            coupled_extension_config=dict(resolved={'population_multiplier':2},sha256='8'*64),
            retrieval_program_changed_only_in_input_path=False,
            binary_sha256=sha256_file(root/'bin/ca3_retrieval'),pattern_sha256=sha256_file(root/'trainPat.csv'),
            scientific_fidelity_qualified=False,
            retrieval_input_contract=dict(effective_input_binding=self.normalized.name,
                source_path_binding_only=False,fresh_coupled_checkpoint_required=True,
                source_carlsim_seed=10,source_host_srand_seed=3,source_estdp_enabled=True))
        self.rebind_runtime(root,build)
        return root

    def rebind_runtime(self,root,build):
        write(root/'build_report.json',build)
        write(root/'manifest.json',dict(files=[dict(path=name,sha256=sha256_file(root/name))
              for name in ('build_report.json','trainPat.csv','bin/ca3_retrieval')]))

    def arguments(self,root):
        return (root,self.formation,self.form_replay,self.normalized,self.normal_report,self.normal_replay)

    def test_new_circuit_cannot_relabel_legacy_formation_or_normalized_graph(self):
        from .ca3_realization import SourceRealization

        requested = SourceRealization('development-circuit-11', 'noise-01', 'retrieval',
                                      'development', 11, 1001, 2001)
        with self.assertRaisesRegex(ValueError, 'NEW_CIRCUIT_REQUIRES_ITS_OWN_FORMATION'):
            self.qualify(source_realization=requested)
        legacy = SourceRealization('normalized-' + sha256_file(self.normalized), 'noise-01',
                                   'retrieval', 'development', 10, 1001, 2001)
        binding = self.qualify(source_realization=legacy)
        self.assertTrue(binding['circuit_identity']['legacy_content_addressed_graph'])

    def test_retrieval_binds_explicit_formation_and_its_prefix_replay(self):
        from dataclasses import replace
        from .ca3_realization import SourceRealization, contract

        formed = SourceRealization('development-circuit-11', 'formation-source', 'formation',
                                   'development', 11, 11, 1)
        requested = SourceRealization(formed.circuit_id, 'noise-01', 'retrieval',
                                      'development', 11, 1001, 2001)
        native = json.loads(self.native_path.read_text())
        native.update(source_realization=contract(formed), checkpoint_sha256='f'*64,
            realization_circuit_binding=dict(circuit_id=formed.circuit_id, circuit_seed=11,
                fresh_native_graph_constructed=True, formation_checkpoint_sha256='f'*64))
        write(self.native_path, native)
        replay = json.loads(self.form_replay.read_text())
        replay.update(source_realization=contract(formed), formation_report_sha256=sha256_file(self.native_path))
        write(self.form_replay, replay)
        binding = self.qualify(source_realization=requested)
        self.assertEqual(binding['circuit_identity']['circuit_seed'], 11)
        self.assertEqual(binding['raw65p_sha256'], sha256_file(self.raw))
        self.assertEqual(binding['normalized_input_sha256'], sha256_file(self.normalized))
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_RETRIEVAL_REALIZATION'):
            self.qualify()
        for changed in [replace(requested, circuit_seed=12), replace(requested, circuit_id='other-circuit')]:
            with self.assertRaisesRegex(ValueError, 'CIRCUIT_IDENTITY_DIFFERS'):
                self.qualify(source_realization=changed)
        replay['source_realization'] = contract(replace(formed, circuit_seed=12, uniform_seed=12))
        write(self.form_replay, replay)
        with self.assertRaisesRegex(ValueError, 'CIRCUIT_IDENTITY_DIFFERS'):
            self.qualify(source_realization=requested)

    def test_valid_coupled_input_preserves_native_normalization_protocol(self):
        binding=self.qualify()
        self.assertTrue(binding['normalized_input_qualified'])
        self.assertEqual(sum(binding['populations'].values()),168106)
        self.assertEqual(len(binding['populations']),8)
        self.assertEqual(binding['coupled_extension_config_sha256'],'8'*64)
        self.assertEqual(binding['normalized_input_sha256'],sha256_file(self.normalized))
        self.assertEqual(json.loads(self.normal_report.read_text())['status'],'APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION')
        self.assertEqual(json.loads(self.normal_replay.read_text())['target_global_mean'],.625)
        with self.assertRaisesRegex(ValueError,'COMPLETE_REPLAYED_FORMATION_REQUIRED'):
            self.qualify(coupled_extension=False)

    def test_old_reference_wrong_group_and_wrong_clock_cannot_be_rebound_as_coupled(self):
        original=self.normalized.read_bytes()
        wrong_group=bytearray(original)
        begin,end=NETWORK_HEADER.size,NETWORK_HEADER.size+GROUP_HEADER.size
        fields=list(GROUP_HEADER.unpack(wrong_group[begin:end]));fields[-1]=b'wrong_population'
        wrong_group[begin:end]=GROUP_HEADER.pack(*fields)
        wrong_clock=bytearray(original);wrong_clock[8:12]=struct.pack('<f',46.)
        for label,payload in [('old84053',self.old_reference_input.read_bytes()),
                              ('wrong_group',bytes(wrong_group)),('wrong_clock',bytes(wrong_clock))]:
            with self.subTest(label=label):
                self.normalized.write_bytes(payload)
                self.rebind_normalization()
                with self.assertRaisesRegex(ValueError,'NORMALIZED65P_IDENTITY_REQUIRED'):
                    self.qualify()
        self.normalized.write_bytes(original)
        self.rebind_normalization()

    def test_formation_replay_must_bind_the_complete_formation_extension(self):
        replay=json.loads(self.form_replay.read_text())
        replay['coupled_extension']['source_pattern_sha256']='7'*64
        write(self.form_replay,replay)
        with self.assertRaisesRegex(ValueError,'FORMATION_REPLAY_COUPLED_IDENTITY_DIFFERS'):
            self.qualify()

    def test_default_execution_and_mismatched_config_do_not_start_a_process(self):
        root=self.runtime()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(execution,'probe') as probe,patch.object(execution.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'PATH_ONLY_RETRIEVAL_BUILD_REQUIRED'):
                execution.execute(*self.arguments(root))
            build=json.loads((root/'build_report.json').read_text())
            build['coupled_extension_config']['sha256']='7'*64
            self.rebind_runtime(root,build)
            with self.assertRaisesRegex(ValueError,'FORMATION_RETRIEVAL_COUPLED_CONFIG_DIFFERS'):
                execution.execute(*self.arguments(root),coupled_extension=True)
            probe.assert_not_called();launch.assert_not_called()
        self.assertFalse((root/'native_run').exists())

    def test_fresh_coupled_checkpoint_flag_is_required(self):
        root=self.runtime()
        original=json.loads((root/'build_report.json').read_text())
        for flag in (None,False):
            build=json.loads(json.dumps(original))
            if flag is None:
                del build['retrieval_input_contract']['fresh_coupled_checkpoint_required']
            else:
                build['retrieval_input_contract']['fresh_coupled_checkpoint_required']=flag
            self.rebind_runtime(root,build)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
                 patch.object(execution,'probe') as probe,patch.object(execution.subprocess,'Popen') as launch:
                with self.subTest(flag=flag),self.assertRaises(ValueError):
                    execution.execute(*self.arguments(root),coupled_extension=True)
                probe.assert_not_called();launch.assert_not_called()
        self.assertFalse((root/'native_run').exists())

    def test_real_false_exit_keeps_coupled_failure_and_never_decodes(self):
        root=self.runtime()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(execution,'probe',return_value={'status':'PASS'}) as probe, \
             patch.object(execution,'validate_retrieval') as decoder:
            with self.assertRaisesRegex(RuntimeError,'RETRIEVAL_EXIT:1'):
                execution.execute(*self.arguments(root),coupled_extension=True)
            decoder.assert_not_called()
        self.assertEqual(probe.call_args.kwargs['required_memory_gib'],128)
        report=json.loads((root/'native_report.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['exit_code'],1)
        self.assertFalse(report['retrieval_executed'])
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertEqual(report['coupled_extension']['source_role'],'retrieval')
        self.assertEqual(report['coupled_extension']['derived_cue_size'],276)
        self.assertEqual((root/'native_run/network65p_assem550_divnorm.dat').resolve(),self.normalized.resolve())
        self.assertFalse((root/'native_run/network65p_assem275_divnorm.dat').exists())
        self.assertFalse((root/'observations').exists())


if __name__ == '__main__':
    unittest.main()


# Caller-only tests use the frozen source package supplied by the test job.
import ast
import hashlib
import xml.etree.ElementTree as ET


def retrieval_caller_namespace(name):
    path=Path(__file__).resolve().parents[2]/'Output/neural_population_dynamics/operations'/name
    body=path.read_text().split("<<'PYTHON'\n",1)[1].rsplit('\nPYTHON',1)[0]
    tree=ast.parse(body)
    names={'sha','validate_realization_source','copy','digest','declared_rows','load_delivery','stage','analysis_placement'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names
           or isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id.startswith('REALIZATION_') for target in node.targets)]
    namespace=dict(Path=Path,json=json,hashlib=hashlib,ET=ET,shutil=shutil)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    namespace['caller_tree']=tree
    return namespace


class RetrievalAnalysisPlacement(unittest.TestCase):
    def setUp(self):
        self.ns=retrieval_caller_namespace('ca3_coupled_retrieval_analysis.sbatch')
        self.placement=self.ns['analysis_placement']

    def environment(self,node='cpu03',partition='gpu',**extra):
        return dict(SLURMD_NODENAME=node,SLURM_JOB_PARTITION=partition,**extra)

    def test_cpu03_route_accepts_both_analysis_modes(self):
        for observation in (False,True):
            with self.subTest(observation=observation):
                self.assertEqual(self.placement(self.environment(),observation=observation),
                                 dict(node='cpu03',partition='gpu'))

    def test_cpu04_accepts_only_observation_readback(self):
        environment=self.environment('cpu04','cpu')
        self.assertEqual(self.placement(environment,observation=True),
                         dict(node='cpu04',partition='cpu'))
        with self.assertRaisesRegex(RuntimeError,'PLACEMENT_REQUIRED'):
            self.placement(environment,observation=False)

    def test_gpu_and_unsupported_placements_reject(self):
        environments=[self.environment(SLURM_JOB_GPUS='0'),
                      self.environment(CUDA_VISIBLE_DEVICES='0'),
                      self.environment('cpu03','cpu'),self.environment('cpu04','gpu'),
                      self.environment('gpu03','gpu1'),{}]
        for environment in environments:
            with self.subTest(environment=environment):
                with self.assertRaises(RuntimeError):
                    self.placement(environment,observation=True)

    def test_delivery_records_actual_placement(self):
        from datetime import datetime,timezone
        tree=self.ns['caller_tree']
        record=next(node for node in tree.body if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='record'
                            for target in node.targets))
        for node,partition in [('cpu03','gpu'),('cpu04','cpu')]:
            with self.subTest(node=node):
                namespace=dict(datetime=datetime,timezone=timezone,job='fixture',
                    placement=self.placement(self.environment(node,partition),observation=True),
                    primary=Path('/primary'),repeat=Path('/observed'),software=Path('/software'),
                    observation=True,native_reference=False)
                exec(compile(ast.Module(body=[record],type_ignores=[]),'delivery_record','exec'),namespace)
                self.assertEqual(namespace['record']['node'],node)
                self.assertEqual(namespace['record']['partition'],partition)
                self.assertEqual(namespace['record']['production_episodes'],0)
                self.assertFalse(namespace['record']['scientific_fidelity_qualified'])


class RetrievalCallerAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ns=retrieval_caller_namespace('ca3_coupled_retrieval_analysis.sbatch')

    def fixture(self,merged=True):
        root=self.root/'validation';root.mkdir()
        (root/'source.py').write_text('qualified_source = True\n')
        def junit(path,number):
            path.write_text('<testsuites><testsuite tests="'+str(number)+'" failures="0" errors="0" skipped="0">'+''.join('<testcase name="case'+str(i)+'"/>' for i in range(number))+'</testsuite></testsuites>')
        junit(root/'junit.xml',2)
        counts=dict(tests=2,failures=0,errors=0,skipped=0)
        value=dict(job_id=603984 if merged else 603229,status='PASS_SOURCE_AND_CALLER_TESTS',exit_code=0,copied_files_verified=True,source_snapshot_isolated_on_compute=True,junit=counts,files=[dict(path='source.py',bytes=(root/'source.py').stat().st_size,sha256=sha256_file(root/'source.py'))])
        accepted=dict(job_id=str(value['job_id']),junit_sha256=sha256_file(root/'junit.xml'),tests=2,testcases=2)
        if merged:
            value.update(tests_rerun=False,previous_validation=603926,prior_test_bodies_preserved=True,verification_runs=[])
            accepted['verification_runs']={}
            for job in (603926,603953):
                prior=self.root/str(job);prior.mkdir();junit(prior/'junit.xml',1)
                previous=dict(job_id=job,status='PASS_SOURCE_AND_CALLER_TESTS' if job==603926 else 'PASS_BACKEND_CONFIGURATION_BINDING',exit_code=0,copied_files_verified=True,junit=dict(tests=1,failures=0,errors=0,skipped=0))
                write(prior/'delivery.json',previous)
                spec=dict(delivery_sha256=sha256_file(prior/'delivery.json'),junit_sha256=sha256_file(prior/'junit.xml'),tests=1,testcases=1,status=previous['status'])
                accepted['verification_runs'][str(job)]=spec
                value['verification_runs'].append(dict(job_id=job,delivery=str(prior/'delivery.json'),delivery_sha256=spec['delivery_sha256'],junit=str(prior/'junit.xml'),junit_sha256=spec['junit_sha256'],counts=previous['junit'],reused=True))
        self.value=value;self.accepted=accepted;self.validation=root
        self.rebind()
        return root

    def rebind(self):
        write(self.validation/'delivery.json',self.value)
        self.ns['REALIZATION_ACCEPTED_VALIDATIONS']={sha256_file(self.validation/'delivery.json'):self.accepted}

    def validate(self):return self.ns['validate_realization_source'](self.validation)

    def test_legacy_exact_binding_remains_supported(self):
        self.fixture(merged=False)
        self.assertEqual(self.validate()[0]['job_id'],603229)

    def test_composed_source_and_supplement_are_bound(self):
        self.fixture()
        self.assertEqual(self.validate()[0]['job_id'],603984)

    def test_foreign_or_modified_receipt_is_rejected(self):
        root=self.fixture();self.value['job_id']=603985;write(root/'delivery.json',self.value)
        with self.assertRaisesRegex(ValueError,'VALIDATED_RECEIPT_REQUIRED'):self.validate()

    def test_tampered_junit_is_rejected(self):
        root=self.fixture();(root/'junit.xml').write_text('<testsuites/>')
        with self.assertRaisesRegex(ValueError,'VALIDATED_JUNIT_REQUIRED'):self.validate()

    def test_tampered_source_is_rejected(self):
        root=self.fixture();(root/'source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'VALIDATION_SOURCE_DIFFERS'):self.validate()

    def test_testcase_count_is_independently_checked(self):
        root=self.fixture();text=(root/'junit.xml').read_text();(root/'junit.xml').write_text(text.replace('<testcase name="case1"/>',''))
        self.accepted['junit_sha256']=sha256_file(root/'junit.xml');self.rebind()
        with self.assertRaisesRegex(ValueError,'SOURCE_TESTS_INCOMPLETE'):self.validate()

    def test_hidden_failure_element_is_rejected(self):
        root=self.fixture();text=(root/'junit.xml').read_text();(root/'junit.xml').write_text(text.replace('<testcase name="case1"/>','<testcase name="case1"><failure/></testcase>'))
        self.accepted['junit_sha256']=sha256_file(root/'junit.xml');self.rebind()
        with self.assertRaisesRegex(ValueError,'SOURCE_TESTS_INCOMPLETE'):self.validate()

    def test_missing_merge_run_is_rejected(self):
        self.fixture();self.value['verification_runs'].pop();self.rebind()
        with self.assertRaisesRegex(ValueError,'MERGED_TEST_PROVENANCE_REQUIRED'):self.validate()

    def test_changed_merged_receipt_is_rejected(self):
        self.fixture();(self.root/'603926/delivery.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'MERGED_TEST_SOURCE_DIFFERS'):self.validate()

    def test_changed_merged_junit_is_rejected(self):
        self.fixture();(self.root/'603953/junit.xml').write_text('<testsuites/>')
        with self.assertRaisesRegex(ValueError,'MERGED_TEST_SOURCE_DIFFERS'):self.validate()


class RetrievalCallerStaging(unittest.TestCase):
    def setUp(self):
        self.fixture=CoupledRetrievalInput('test_valid_coupled_input_preserves_native_normalization_protocol')
        self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.ns=retrieval_caller_namespace('ca3_coupled_retrieval.sbatch')
        self.runtime=self.fixture.root/'caller_runtime';self.runtime.mkdir()
        self.staged=self.runtime/'formation_inputs'
        self.ns.update(runtime=self.runtime,formation=self.fixture.formation,staged_formation=self.staged,plan=[],receipt=dict(input_deliveries={},staged_inputs=[]))

    def formation(self):
        from .ca3_realization import SourceRealization,contract
        formed=SourceRealization('development-circuit-11','formation-source','formation','development',11,11,1)
        requested=SourceRealization(formed.circuit_id,'noise-01','retrieval','development',11,1001,2001)
        native=json.loads(self.fixture.native_path.read_text())
        native.update(source_realization=contract(formed),checkpoint_sha256='f'*64,realization_circuit_binding=dict(circuit_id=formed.circuit_id,circuit_seed=11,fresh_native_graph_constructed=True,formation_checkpoint_sha256='f'*64))
        write(self.fixture.native_path,native)
        replay=json.loads(self.fixture.form_replay.read_text());replay.update(source_realization=contract(formed),formation_report_sha256=sha256_file(self.fixture.native_path))
        write(self.fixture.form_replay,replay)
        return requested

    def declare(self):
        names=('native_report.json','observations/recording.json','native_run/network65p.dat')
        write(self.fixture.formation/'delivery.json',dict(status='COUPLED_FORMATION_AND_NATIVE_RECORDS_VALIDATED',exit_code=0,copied_files_verified=True,files=[dict(path=name,bytes=(self.fixture.formation/name).stat().st_size,sha256=sha256_file(self.fixture.formation/name)) for name in names]))
        self.ns['formation_rows']=self.ns['load_delivery'](self.fixture.formation,'formation','COUPLED_FORMATION_AND_NATIVE_RECORDS_VALIDATED')

    def stage(self):
        loops=[node for node in ast.walk(self.ns['caller_tree']) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='relative' and isinstance(node.iter,ast.Tuple) and [getattr(x,'value',None) for x in node.iter.elts]==['native_report.json','observations/recording.json','native_run/network65p.dat']]
        self.assertEqual(len(loops),1)
        exec(compile(ast.Module(body=loops,type_ignores=[]),'actual_retrieval_formation_plan','exec'),self.ns)
        for root,relative,target,row,label in self.ns['plan']:self.ns['stage'](root,relative,target,row,label)

    def qualify(self,requested):
        f=self.fixture
        return execution.qualify_input(self.staged,f.form_replay,f.normalized,f.normal_report,f.normal_replay,coupled_extension=True,source_realization=requested)

    def test_seed11_staging_reaches_frozen_input_qualification(self):
        requested=self.formation();self.declare();self.stage()
        self.assertFalse((self.staged/'realization_source.cpp').exists())
        self.assertFalse((self.staged/'source_realization.yaml').exists())
        self.assertEqual(self.qualify(requested)['circuit_identity']['circuit_seed'],11)
        self.assertEqual(len(self.ns['receipt']['staged_inputs']),3)

    def test_legacy_formation_staging_remains_accepted(self):
        self.declare();self.stage()
        self.assertTrue(self.qualify(None)['normalized_input_qualified'])

    def test_tampered_formation_is_rejected_by_actual_stager(self):
        self.formation();self.declare();self.fixture.raw.write_bytes(self.fixture.raw.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'STAGED_FILE_DIFFERS'):self.stage()

    def test_foreign_circuit_is_rejected_after_staging(self):
        from dataclasses import replace
        requested=self.formation();self.declare();self.stage()
        with self.assertRaisesRegex(ValueError,'CIRCUIT_IDENTITY_DIFFERS'):self.qualify(replace(requested,circuit_seed=12))


class RetrievalAnalysisCapsuleStaging(unittest.TestCase):
    def setUp(self):
        self.fixture=RetrievalCallerAcceptance()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.ns=self.fixture.ns
        tree=self.ns['caller_tree']
        bindings=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                  and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                  and node.value.func.id=='validate_realization_source']
        self.assertEqual(len(bindings),1)
        def is_receipt_copy(node,name):
            return (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Name) and node.value.func.id=='copy'
                    and len(node.value.args)==3 and isinstance(node.value.args[1],ast.Constant)
                    and node.value.args[1].value==name)
        stages=[node for node in ast.walk(tree) if isinstance(node,ast.If)
                and any(is_receipt_copy(child,'source_validation_junit.xml') for child in node.body)]
        self.assertEqual(len(stages),1)
        copies=[node for node in stages[0].body if any(is_receipt_copy(node,name) for name in
                ('source_validation_delivery.json','source_validation_junit.xml'))]
        self.assertEqual(len(copies),2)
        self.binding=compile(ast.Module(body=bindings,type_ignores=[]),'actual_source_binding','exec')
        self.copies=compile(ast.Module(body=copies,type_ignores=[]),'actual_source_receipt_staging','exec')

    def prepare(self,merged=True):
        source=self.fixture.fixture(merged=merged)
        self.ns.update(source_validation=source,validation_path=source/'delivery.json',
                       work=self.fixture.root/'staged',outputs=set())
        exec(self.binding,self.ns)
        return source

    def copy_receipts(self):
        exec(self.copies,self.ns)

    def assert_staged_receipts(self,merged):
        source=self.prepare(merged=merged);self.copy_receipts()
        for original,staged in (('delivery.json','source_validation_delivery.json'),
                                ('junit.xml','source_validation_junit.xml')):
            self.assertEqual((source/original).read_bytes(),(self.ns['work']/staged).read_bytes())
        self.assertEqual(self.ns['realization_pins']['delivery_sha256'],sha256_file(source/'delivery.json'))
        self.assertEqual(self.ns['realization_pins']['junit_sha256'],self.fixture.accepted['junit_sha256'])

    def test_legacy_603229_receipts_stage_from_their_accepted_pins(self):
        self.assert_staged_receipts(merged=False)

    def test_composed_603984_receipts_stage_from_their_accepted_pins(self):
        self.assert_staged_receipts(merged=True)

    def test_receipt_changed_after_validation_is_rejected_by_actual_copy(self):
        source=self.prepare();(source/'delivery.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'STAGED_CODE_DIFFERS:source_validation_delivery.json'):
            self.copy_receipts()

    def test_junit_changed_after_validation_is_rejected_by_actual_copy(self):
        source=self.prepare();(source/'junit.xml').write_text('<testsuites/>')
        with self.assertRaisesRegex(ValueError,'STAGED_CODE_DIFFERS:source_validation_junit.xml'):
            self.copy_receipts()

    def test_wrong_capsule_identity_is_rejected_before_staging(self):
        source=self.fixture.fixture();self.fixture.value['job_id']=603229;self.fixture.rebind()
        self.ns.update(source_validation=source)
        with self.assertRaisesRegex(ValueError,'PASSED_SOURCE_VALIDATION_REQUIRED'):
            exec(self.binding,self.ns)
