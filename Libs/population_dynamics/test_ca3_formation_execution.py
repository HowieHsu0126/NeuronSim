import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import ca3_formation_execution as formation
from .provenance import sha256_file
from .test_carlsim_formation_replay import coupled_declaration
from types import SimpleNamespace


class NativeExecutionFailure(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        (self.root/'bin').mkdir()
        shutil.copy2('/bin/false', self.root/'bin/ca3_formation')
        (self.root/'trainPat.csv').write_text('0 1 2\n')
        report = dict(scope='native_source_formation',status='NATIVE_SOURCE_BUILD_PASSED_GPU_RUN_PENDING',
                      complete_author_formation_program_unchanged=True,expected_duration_ms=70000,
                      core_neurons=84053,overlay_source_sha256='a'*64,original_author_main_sha256='a'*64,
                      binary_sha256=sha256_file(self.root/'bin/ca3_formation'))
        (self.root/'build_report.json').write_text(json.dumps(report))
        paths=['bin/ca3_formation','trainPat.csv','build_report.json']
        manifest=dict(reference_checkpoint_bytes=1,files=[dict(path=p,sha256=sha256_file(self.root/p)) for p in paths])
        (self.root/'manifest.json').write_text(json.dumps(manifest))

    def test_allocation_and_input_identity_fail_before_launch(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(ValueError,'ALLOCATED_GPU'):
                formation.execute(self.root)
        self.fixture()
        (self.root/'trainPat.csv').write_text('changed')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'INPUT_CHANGED'):
                formation.execute(self.root)
        self.assertFalse((self.root/'native_run').exists())

    def test_failed_process_is_recorded_without_decoder_or_fidelity_promotion(self):
        self.fixture()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}), \
             patch.object(formation,'probe',return_value={'status':'PASS'}), \
             patch.object(formation,'validate_formation') as decoder:
            with self.assertRaisesRegex(RuntimeError,'FORMATION_EXIT:1'):
                formation.execute(self.root)
            decoder.assert_not_called()
        report=json.loads((self.root/'native_report.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['exit_code'],1)
        self.assertFalse(report['complete_formation_executed'])
        self.assertTrue((self.root/'native_run/trainPat.csv').is_file())
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaises(FileExistsError):
                formation.execute(self.root)


class NativePrefixExecution(unittest.TestCase):
    """Exercise real executor boundaries with inert CPU binaries, never CARLsim."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'bin').mkdir()
        shutil.copy2('/bin/false', self.root / 'bin/ca3_prefix')
        (self.root / 'trainPat.csv').write_text('0 1 2\n')
        self.build = dict(scope='native_source_prefix', status='NATIVE_SOURCE_BUILD_PASSED_GPU_RUN_PENDING',
            complete_author_formation_program_unchanged=False, expected_duration_ms=4450,
            core_neurons=84053, overlay_source_sha256='b'*64, original_author_main_sha256='a'*64,
            source_code_semantics_preserved_before_stop=True,
            binary_sha256=sha256_file(self.root / 'bin/ca3_prefix'), scientific_fidelity_qualified=False)

    def tearDown(self):
        self.temp.cleanup()

    def write_build(self):
        (self.root / 'build_report.json').write_text(json.dumps(self.build))
        paths = ('bin/ca3_prefix', 'trainPat.csv', 'build_report.json')
        (self.root / 'manifest.json').write_text(json.dumps(dict(reference_checkpoint_bytes=128,
            files=[dict(path=name, sha256=sha256_file(self.root/name)) for name in paths])))

    def corrected_build(self):
        from .ca3_conductance_memory import GPU_FILE, OVERLAY_ID, PINNED_SHA256

        self.build.update(author_program_semantics_preserved_before_stop=True,
            source_code_semantics_preserved_before_stop=False,
            conductance_memory_implementation_sha256=sha256_file(
                Path(formation.__file__).with_name('ca3_conductance_memory.py')),
            conductance_memory_overlay=dict(overlay_id=OVERLAY_ID,
                original_sha256={GPU_FILE: PINNED_SHA256}, input_sha256={GPU_FILE: PINNED_SHA256},
                overlay_sha256={GPU_FILE: 'c'*64}, kernel_arithmetic_unchanged=True,
                precision_unchanged=True, native_qualification=False,
                initialization_correction=dict(corrected_extent='syn_gPitch * numNReg',
                    previously_uninitialized_rows_are_now_zeroed=True)))

    def assert_process_failure_retained(self):
        self.write_build()
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
             patch.object(formation, 'probe', return_value={'status':'PASS'}), \
             patch.object(formation.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024**4)), \
             patch.object(formation, 'validate_prefix') as decoder:
            with self.assertRaisesRegex(RuntimeError, 'PREFIX_EXIT:1'):
                formation.execute(self.root, scope='prefix')
            decoder.assert_not_called()
        report = json.loads((self.root/'native_report.json').read_text())
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['exit_code'], 1)
        self.assertFalse(report['complete_formation_executed'])
        self.assertEqual(report['source_code_semantics_preserved_before_stop'],
                         self.build['source_code_semantics_preserved_before_stop'])

    def test_legacy_native_prefix_without_author_only_field_remains_accepted(self):
        self.assertNotIn('author_program_semantics_preserved_before_stop', self.build)
        self.assert_process_failure_retained()

    def test_corrected_native_prefix_preserves_real_process_failure(self):
        self.corrected_build()
        self.assert_process_failure_retained()

    def test_corrected_native_prefix_dispatch_preserves_counts_clock_and_truthful_flags(self):
        self.corrected_build()
        shutil.copy2('/bin/true', self.root/'bin/ca3_prefix')
        self.build['binary_sha256'] = sha256_file(self.root/'bin/ca3_prefix')
        self.write_build()

        def checkpoint_fixture(run, scope, *, expected_units):
            self.assertEqual(scope, 'prefix')
            self.assertEqual(expected_units, 84053)
            (run/'network5p.dat').write_bytes(b'DISPATCH FIXTURE; NOT NEURAL STATE')
            return [dict(fixture_only=True, units=84053)]

        with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
             patch.object(formation, 'probe', return_value={'status':'PASS'}) as probe, \
             patch.object(formation.shutil, 'disk_usage', return_value=SimpleNamespace(free=8*1024**3+128)), \
             patch.object(formation, 'verify_checkpoints', side_effect=checkpoint_fixture), \
             patch.object(formation, 'validate_prefix') as decoder, \
             patch.object(formation, 'validate_formation') as full_decoder:
            report = formation.execute(self.root, scope='prefix')
            decoder.assert_called_once_with(self.root, self.root/'observations', coupled_extension=False)
            full_decoder.assert_not_called()
        probe.assert_called_once_with(self.root/'resource_preflight.json', required_memory_gib=64)
        self.assertEqual(report['status'], 'NATIVE_PREFIX_EXIT_ZERO')
        self.assertTrue(report['author_program_semantics_preserved_before_stop'])
        self.assertFalse(report['source_code_semantics_preserved_before_stop'])
        self.assertFalse(report['complete_formation_executed'])
        self.assertFalse(report['complete_formation_retrieval'])
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertEqual(report['conductance_memory_overlay'], self.build['conductance_memory_overlay'])

    def test_invalid_native_prefix_proofs_fail_before_resource_check_or_launch(self):
        self.corrected_build()
        valid = json.loads(json.dumps(self.build))
        changes = {
            'missing_author_proof': lambda r: r.pop('author_program_semantics_preserved_before_stop'),
            'false_author_proof': lambda r: r.update(author_program_semantics_preserved_before_stop=False),
            'missing_correction': lambda r: r.pop('conductance_memory_overlay'),
            'false_global_preservation': lambda r: r.update(source_code_semantics_preserved_before_stop=True),
            'unknown_backend': lambda r: r['conductance_memory_overlay'].update(overlay_id='unknown'),
            'uncorrected_initialization': lambda r: r['conductance_memory_overlay']['initialization_correction'].update(
                previously_uninitialized_rows_are_now_zeroed=False),
            'changed_helper': lambda r: r.update(conductance_memory_implementation_sha256='d'*64),
            'wrong_count': lambda r: r.update(core_neurons=168106),
            'wrong_clock': lambda r: r.update(expected_duration_ms=4451),
            'no_prefix_stop': lambda r: r.update(overlay_source_sha256=r['original_author_main_sha256']),
        }
        for name, change in changes.items():
            self.build = json.loads(json.dumps(valid))
            change(self.build)
            self.write_build()
            with self.subTest(change=name), \
                 patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
                 patch.object(formation, 'probe') as probe, \
                 patch.object(formation.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'CA3_NATIVE_PREFIX_BUILD_REQUIRED'):
                    formation.execute(self.root, scope='prefix')
                probe.assert_not_called()
                launch.assert_not_called()
        self.assertFalse((self.root/'native_report.json').exists())
        self.assertFalse((self.root/'native_run').exists())


class CoupledFormationExecution(unittest.TestCase):
    """Executor fixtures only: no GPU work or scientific simulation is performed."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'bin').mkdir()
        shutil.copy2('/bin/false',self.root/'bin/ca3_formation')
        (self.root/'trainPat.csv').write_text(' '.join(str(i) for i in list(range(550))*4)+'\n')
        extension=coupled_declaration()
        extension['derived_pattern_sha256']=sha256_file(self.root/'trainPat.csv')
        self.build=dict(scope='coupled_extension_formation',
            status='COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED',
            complete_author_formation_program_unchanged=False,expected_duration_ms=70000,
            core_neurons=168106,populations=extension['derived_populations'],
            source_code_semantics_preserved_before_stop=False,coupled_extension=extension,
            coupled_extension_config=dict(resolved={'population_multiplier':2},sha256='8'*64),
            original_author_main_sha256='a'*64,overlay_source_sha256='b'*64,
            binary_sha256=sha256_file(self.root/'bin/ca3_formation'),
            scientific_fidelity_qualified=False,full_scale_qualified=False)
        self.write_build(self.build)

    def tearDown(self):
        self.temp.cleanup()

    def write_build(self,report,binary_name="ca3_formation"):
        (self.root/'build_report.json').write_text(json.dumps(report))
        paths=('bin/'+binary_name,'trainPat.csv','build_report.json')
        (self.root/'manifest.json').write_text(json.dumps(dict(reference_checkpoint_bytes=128,
            files=[dict(path=name,sha256=sha256_file(self.root/name)) for name in paths])))

    def test_derived_build_requires_explicit_opt_in(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe') as probe,patch.object(formation.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'UNCHANGED_FULL_FORMATION_BUILD_REQUIRED'):
                formation.execute(self.root)
            probe.assert_not_called();launch.assert_not_called()
        self.assertFalse((self.root/'native_report.json').exists())
        self.assertFalse((self.root/'native_run').exists())

    def test_invalid_coupled_identity_and_false_source_claims_rejected(self):
        for change in ('reference_count','cloned_graph','unchanged_source'):
            report=json.loads(json.dumps(self.build))
            if change=='reference_count':
                report['core_neurons']=84053
            elif change=='cloned_graph':
                report['coupled_extension']['graph_or_output_cloning']=True
            else:
                report['complete_author_formation_program_unchanged']=True
                report['overlay_source_sha256']=report['original_author_main_sha256']
            self.write_build(report)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
                 patch.object(formation,'probe') as probe,patch.object(formation.subprocess,'Popen') as launch:
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'CA3_COUPLED_'):
                    formation.execute(self.root,coupled_extension=True)
                probe.assert_not_called();launch.assert_not_called()
        self.assertFalse((self.root/'native_run').exists())

    def test_340_gib_admission_failure_is_retained_without_launch(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe',return_value={'status':'BLOCKED'}) as probe, \
             patch.object(formation.subprocess,'Popen') as launch, \
             patch.object(formation,'validate_formation') as decoder:
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_PREFLIGHT_FAILED'):
                formation.execute(self.root,coupled_extension=True)
            launch.assert_not_called();decoder.assert_not_called()
        probe.assert_called_once_with(self.root/'resource_preflight.json',required_memory_gib=340)
        report=json.loads((self.root/'native_report.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['resource_admission'],dict(
            minimum_memory_gib=340,reference_population_multiplier=2,extrapolated=True))
        self.assertFalse(report['native_run_executed'])
        self.assertFalse(report['complete_formation_executed'])
        self.assertFalse((self.root/'native_run').exists())

    def test_real_false_exit_records_failure_without_checkpoint_or_decoder(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe',return_value={'status':'PASS'}) as probe, \
             patch.object(formation.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)), \
             patch.object(formation,'verify_checkpoints') as checkpoints, \
             patch.object(formation,'validate_formation') as decoder:
            with self.assertRaisesRegex(RuntimeError,'FORMATION_EXIT:1'):
                formation.execute(self.root,coupled_extension=True)
            checkpoints.assert_not_called();decoder.assert_not_called()
        self.assertEqual(probe.call_args.kwargs['required_memory_gib'],340)
        report=json.loads((self.root/'native_report.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['exit_code'],1)
        self.assertTrue(report['native_run_executed'])
        self.assertFalse(report['complete_formation_executed'])
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertFalse(report['full_scale_qualified'])
        self.assertEqual(report['coupled_extension']['derived_core_neurons'],168106)
        self.assertEqual((self.root/'native_run/trainPat.csv').read_bytes(),(self.root/'trainPat.csv').read_bytes())
        self.assertFalse((self.root/'observations').exists())

    def test_success_dispatch_uses_derived_checkpoint_count_and_decoder_flag(self):
        # /bin/true supplies exit0 only. Checkpoint/decoder mocks deliberately do
        # not claim that CARLsim ran or that the checkpoint contains neural state.
        shutil.copy2('/bin/true',self.root/'bin/ca3_formation')
        self.build['binary_sha256']=sha256_file(self.root/'bin/ca3_formation')
        self.write_build(self.build)
        marker=[dict(fixture_only=True,units=168106)]
        def checkpoint_fixture(run,scope,*,expected_units):
            self.assertEqual(run,self.root/'native_run')
            self.assertEqual(scope,'full-formation')
            self.assertEqual(expected_units,168106)
            (run/'network100p.dat').write_bytes(b'DISPATCH FIXTURE ONLY; NOT A SCIENTIFIC CHECKPOINT')
            return marker
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe',return_value={'status':'PASS'}), \
             patch.object(formation.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)), \
             patch.object(formation,'verify_checkpoints',side_effect=checkpoint_fixture) as checkpoints, \
             patch.object(formation,'validate_formation') as decoder:
            report=formation.execute(self.root,coupled_extension=True)
            checkpoints.assert_called_once_with(self.root/'native_run','full-formation',expected_units=168106)
            decoder.assert_called_once_with(self.root,self.root/'observations',coupled_extension=True)
        self.assertEqual(report['status'],'COUPLED_EXTENSION_FORMATION_EXIT_ZERO')
        self.assertEqual(report['checkpoints'],marker)
        self.assertTrue(report['complete_formation_executed'])
        self.assertFalse(report['complete_formation_retrieval'])
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertFalse(report['full_scale_qualified'])


    def prepare_prefix(self, executable='/bin/false'):
        shutil.copy2(executable,self.root/'bin/ca3_prefix')
        self.build.update(scope='coupled_extension_prefix',expected_duration_ms=4450,
                          binary_sha256=sha256_file(self.root/'bin/ca3_prefix'))
        self.write_build(self.build,binary_name='ca3_prefix')

    def test_prefix_requires_explicit_scope_and_exact_clock(self):
        self.prepare_prefix()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe') as probe,patch.object(formation.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'COUPLED_FULL_FORMATION_BUILD_REQUIRED'):
                formation.execute(self.root,coupled_extension=True)
            self.build['expected_duration_ms']=4451
            self.write_build(self.build,binary_name='ca3_prefix')
            with self.assertRaisesRegex(ValueError,'COUPLED_PREFIX_BUILD_REQUIRED'):
                formation.execute(self.root,coupled_extension=True,scope='prefix')
            probe.assert_not_called();launch.assert_not_called()
        self.assertFalse((self.root/'native_report.json').exists())

    def test_prefix_success_uses_one_checkpoint_and_never_claims_full_formation(self):
        self.prepare_prefix('/bin/true')
        def checkpoint_fixture(run,scope,*,expected_units):
            self.assertEqual(scope,'prefix');self.assertEqual(expected_units,168106)
            (run/'network5p.dat').write_bytes(b'PREFIX DISPATCH FIXTURE; NOT NEURAL STATE')
            return [dict(fixture_only=True,units=168106)]
        # Only enough fixture scratch for one scaled checkpoint, not twenty.
        free=8*1024**3+2*128
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe',return_value={'status':'PASS'}) as probe, \
             patch.object(formation.shutil,'disk_usage',return_value=SimpleNamespace(free=free)), \
             patch.object(formation,'verify_checkpoints',side_effect=checkpoint_fixture) as checkpoints, \
             patch.object(formation,'validate_prefix') as decoder, \
             patch.object(formation,'validate_formation') as full_decoder:
            report=formation.execute(self.root,coupled_extension=True,scope='prefix')
            checkpoints.assert_called_once_with(self.root/'native_run','prefix',expected_units=168106)
            decoder.assert_called_once_with(self.root,self.root/'observations',coupled_extension=True)
            full_decoder.assert_not_called()
        probe.assert_called_once_with(self.root/'resource_preflight.json',required_memory_gib=128)
        self.assertEqual(report['status'],'COUPLED_EXTENSION_PREFIX_EXIT_ZERO')
        self.assertEqual(report['checkpoint_relative_path'],'native_run/network5p.dat')
        self.assertFalse(report['complete_formation_executed'])
        self.assertFalse(report['complete_formation_retrieval'])
        self.assertFalse(report['scientific_fidelity_qualified'])

    def test_prefix_process_failure_is_retained_without_decoding(self):
        self.prepare_prefix()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'},clear=True), \
             patch.object(formation,'probe',return_value={'status':'PASS'}), \
             patch.object(formation.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)), \
             patch.object(formation,'validate_prefix') as decoder:
            with self.assertRaisesRegex(RuntimeError,'PREFIX_EXIT:1'):
                formation.execute(self.root,coupled_extension=True,scope='prefix')
            decoder.assert_not_called()
        report=json.loads((self.root/'native_report.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['exit_code'],1)
        self.assertFalse(report['complete_formation_executed'])


if __name__ == '__main__':
    unittest.main()
