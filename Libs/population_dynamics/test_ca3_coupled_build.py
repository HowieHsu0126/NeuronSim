"""CA3 caller contracts with real source and explicitly mocked build dependencies.

These tests never compile or execute CARLsim. The mock compiler product and small
checkpoint-header fixture are software test inputs, not native-build evidence.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import unittest
from unittest.mock import patch

from . import ca3_reference as builder
from . import ca3_conductance_memory as memory_overlay
from . import ca3_formation_execution as formation_runner
from . import ca3_retrieval_execution as retrieval_runner
from . import carlsim_records as records
from .ca3_parameter_comparison import native_parameters
from .provenance import sha256_file
from . import ca3_cue_omission as omission


class CoupledBuildContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        libs = Path(__file__).resolve().parents[1]
        cls.author = Path(os.environ.get("NEURONSIM_CA3_SOURCE_ROOT", str(libs / "external/ca3_formation_retrieval")))
        cls.campaign = Path(os.environ.get("NEURONSIM_CA3_CAMPAIGN_CONFIG", str(libs / "configs/neural_population_dynamics.yaml")))
        cls.original = {}
        for scope in ("full-formation", "retrieval"):
            selected = builder.select_project(scope)
            paths = [selected.project + '/' + name for name in
                     (selected.main, "generateCONFIGStateSTP.h", "generateSETUPStateSTP.h")]
            for relative in [*paths, selected.patterns]:
                path = cls.author / relative
                if not path.is_file():
                    raise RuntimeError("Stage the actual CA3 fixture file: " + str(path))
                cls.original[relative] = path.read_bytes()
        cls.configuration = builder.load_config(cls.campaign)
        if cls.configuration.get("ca3_coupled_extension") != {"population_multiplier": 2}:
            raise RuntimeError("Stage the campaign YAML containing the explicit twofold extension")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ca3-coupled-build-tests-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_originals_preserved(self):
        for relative, original in self.original.items():
            self.assertEqual((self.author / relative).read_bytes(), original, relative)

    def copied_project(self, scope, label):
        selected = builder.select_project(scope)
        project = self.author / selected.project
        local = self.root / label
        for name in (selected.main, "generateCONFIGStateSTP.h", "generateSETUPStateSTP.h"):
            destination = local / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, destination)
        return selected, project, local, self.author / selected.patterns

    def stage_fixture(self, source, target):
        """Copy only declared real author inputs; supply one inert CMake file."""
        target.mkdir(parents=True)
        if source.name == "carlsim_ca3":
            (target / "CMakeLists.txt").write_text("# inert compiler-dependency fixture\n")
            # Explicit realization builds authenticate the real native reset.
            reset_source = Path(os.environ.get('NEURONSIM_CARLSIM_SOURCE_ROOT',
                str(Path(__file__).resolve().parents[1]/'external/carlsim_ca3')))
            reset_relative = 'carlsim/kernel/src/snn_manager.cpp'
            destination = target/reset_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reset_source/reset_relative, destination)
            if getattr(self, 'refinement_framework_fixture', False):
                from .ca3_neuron_substeps import FRAMEWORK_SHA256

                root = Path(os.environ.get('NEURONSIM_CARLSIM_SOURCE_ROOT',
                    str(Path(__file__).resolve().parents[1]/'external/carlsim_ca3')))
                for relative in FRAMEWORK_SHA256:
                    destination = target/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root/relative, destination)
            return {"revision": records.SOURCE_REVISION}
        self.assertEqual(source.name, "ca3_formation_retrieval")
        for relative, content in self.original.items():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return {"revision": retrieval_runner.AUTHOR_REVISION}

    def build_fixture(self, scope, *, coupled, conductance_memory=False, cue_omission=False,
                      neuron_substeps=None, source_realization_config=None, native_observation=None):
        selected = builder.select_project(scope)
        work = self.root / (("derived-" if coupled else "reference-") + scope)
        if neuron_substeps is not None:
            work = work.with_name(work.name + '-' + str(neuron_substeps))

        def compiler_fixture(command, **kwargs):
            self.assertEqual(command[0], "cmake")
            if command[1] == "--build":
                binary = Path(command[2]) / "projects" / selected.local_name / selected.target
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"INERT MOCK COMPILER PRODUCT; NOT A NATIVE BUILD\n")

        with patch.dict(os.environ, {"SLURM_JOB_ID": "software-fixture", "SLURM_CPUS_PER_TASK": "1"}, clear=True), \
             patch.object(builder, "stage", side_effect=self.stage_fixture), \
             patch.object(builder, "materialize_build_inputs", return_value={"compiler_fixture": True}), \
             patch.object(builder.subprocess, "run", side_effect=compiler_fixture) as run, \
             patch.object(builder.subprocess, "check_output", return_value="mock toolchain; not native evidence"):
            result = builder.execute(
                self.root / "declared-staging-inputs", work, "89", scope=scope, build_only=True,
                local_retrieval_input=((coupled or cue_omission) and scope == "retrieval"),
                coupled_extension_config=self.campaign if coupled else None,
                conductance_memory=conductance_memory, cue_omission=cue_omission,
                native_observation=(cue_omission or neuron_substeps is not None or source_realization_config is not None)
                    if native_observation is None else native_observation,
                neuron_substeps=neuron_substeps, source_realization_config=source_realization_config)
            self.assertEqual(run.call_count, 2)
        self.assertFalse(result["native_run_executed"])
        self.assertFalse((work / "native_run").exists())
        self.assertEqual(json.loads((work / "report.json").read_text()), result)
        return result, work

    def test_explicit_circuit_builds_reach_unchanged_formation_resource_gate(self):
        from .ca3_realization import SourceRealization, bind_runtime, contract, verify_formation_program

        spec = SourceRealization('development-circuit-11', 'formation-source', 'formation',
                                 'development', 11, 11, 1)
        config = self.root/'formation-realization.yaml'
        config.write_text(json.dumps({**self.configuration, 'ca3_source_realization': spec.record()}))
        for scope in ('prefix', 'full-formation'):
            with self.subTest(scope=scope), patch.object(memory_overlay, 'apply_overlay', return_value={'fixture': True}):
                result, work = self.build_fixture(scope, coupled=True, conductance_memory=True,
                    source_realization_config=config, native_observation=False)
            selected = builder.select_project(scope)
            main = (work/'carlsim/projects'/selected.local_name/selected.main).read_bytes()
            self.assertIn(b'int randSeed = 11;', main)
            self.assertNotIn(b'Explicit stage-noise realization', main)
            self.assertEqual(result['source_realization'], contract(spec))
            self.assertTrue(result['source_realization']['source_topology_noise_coupling_preserved'])
            verify_formation_program(main, spec, prefix=scope == 'prefix')
            runtime = self.runtime_fixture(result, work, 'circuit-'+scope)
            if scope == 'prefix':
                (runtime/'bin/ca3_formation').rename(runtime/'bin/ca3_prefix')
                manifest = json.loads((runtime/'manifest.json').read_text())
                for row in manifest['files']:
                    if row['path']=='bin/ca3_formation':row['path']='bin/ca3_prefix'
                (runtime/'manifest.json').write_text(json.dumps(manifest))
            (runtime/'realization_source.cpp').write_bytes(main)
            self.assertEqual(bind_runtime(runtime, result, config), spec)
            with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
                 patch.object(formation_runner, 'probe', side_effect=RuntimeError('RESOURCE_GATE_REACHED')) as probe, \
                 patch.object(formation_runner.subprocess, 'Popen') as native:
                with self.assertRaisesRegex(RuntimeError, 'RESOURCE_GATE_REACHED'):
                    formation_runner.execute(runtime, coupled_extension=True, scope=scope,
                                             source_realization_config=config)
                self.assertEqual(probe.call_args.kwargs['required_memory_gib'], 128 if scope=='prefix' else 340)
                native.assert_not_called()
        self.assert_originals_preserved()

    def test_new_circuit_formation_worker_preserves_independent_prefix_and_hash_chain(self):
        import ast
        import struct
        from dataclasses import replace
        from .ca3_realization import SourceRealization, contract
        from .carlsim_normalization import SYNAPSE_DTYPE
        from .test_carlsim_formation_replay import coupled_fixture
        from . import carlsim_formation_replay as formation_replay

        spec = SourceRealization('development-circuit-11', 'formation-source', 'formation',
                                 'development', 11, 11, 1)
        config = self.root/'formation-worker.yaml'
        config.write_text(json.dumps({**self.configuration, 'ca3_source_realization': spec.record()}))
        prefix, full = self.root/'prefix-events', self.root/'formation-events'
        coupled_fixture(prefix); coupled_fixture(full, prefix)
        for scope, root in [('prefix',prefix), ('full-formation',full)]:
            with patch.object(memory_overlay, 'apply_overlay', return_value={'fixture': True}):
                build, work = self.build_fixture(scope, coupled=True, conductance_memory=True,
                    source_realization_config=config, native_observation=False)
            selected = builder.select_project(scope)
            shutil.copy2(work/'carlsim/projects'/selected.local_name/selected.main, root/'realization_source.cpp')
            shutil.copy2(config, root/'source_realization.yaml')
            native = json.loads((root/'native_report.json').read_text())
            for name in ['source_realization','source_realization_config','source_file_byte_checks',
                         'original_author_main_sha256','overlay_source_sha256','source_rng',
                         'author_revision','coupled_extension','coupled_extension_config','pattern_sha256']:
                native[name] = build[name]
            native['realization_circuit_binding'] = dict(circuit_id=spec.circuit_id, circuit_seed=11,
                fresh_native_graph_constructed=True, formation_checkpoint_sha256=native['checkpoint_sha256'])
            (root/'native_report.json').write_text(json.dumps(native))
            shutil.rmtree(root/'observations')
            decoder = records.validate_prefix if scope=='prefix' else records.validate_formation
            decoder(root, root/'observations', coupled_extension=True, source_realization_config=config)
        header = records.network_header(full/'native_run/network100p.dat')
        content = bytearray((full/'native_run/network100p.dat').read_bytes()[:header['payload_offset']])
        struct.pack_into('<f', content, 8, 45.85)
        pyr = next(g.group_id for g in header['groups'] if g.name=='CA3_Pyramidal')
        synapses = np.zeros(4,dtype=SYNAPSE_DTYPE)
        synapses['pre_group']=pyr;synapses['post_group']=pyr
        synapses['pre_unit']=[0,1,2,3];synapses['post_unit']=[1,2,3,4]
        synapses['connection']=[50,50,50,0];synapses['weight']=[.25,.5,.75,1.0]
        synapses['max_weight']=2;synapses['delay']=1
        (full/'native_run/network65p.dat').write_bytes(content+struct.pack('<ii',1,4)+synapses.tobytes())
        for role, root in [('prefix',prefix),('formation',full)]:
            recording = json.loads((root/'observations/recording.json').read_text())
            names=['native_report.json','observations/recording.json','observations/population_counts.npz',
                   'native_run/network5p.dat','source_realization.yaml','realization_source.cpp']
            if role=='formation':names += ['native_run/network65p.dat','native_run/network100p.dat']
            names += [row['relative_path'] for row in recording['files']]
            (root/'delivery.json').write_text(json.dumps(dict(
                status='COUPLED_'+role.upper()+'_AND_NATIVE_RECORDS_VALIDATED',exit_code=0,copied_files_verified=True,
                files=[dict(path=name,bytes=(root/name).stat().st_size,sha256=sha256_file(root/name)) for name in names])))
        repository = Path(__file__).resolve().parents[2]
        launcher = repository/'Output/neural_population_dynamics/operations/ca3_coupled_analysis.sbatch'
        tree=ast.parse(launcher.read_text().split("<<'PYTHON'\n",1)[1].rsplit('\nPYTHON',1)[0])
        worker=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='WORKER' for target in node.targets))
        case=self.root/'worker';case.mkdir();(case/'scratch').mkdir();(case/'analysis_worker.py').write_text(worker)
        env={**os.environ,'PYTHONPATH':str(repository/'Libs'),'TMPDIR':str(case/'scratch'),
             'CA3_NATIVE_REFERENCE':'0','CA3_SOURCE_REALIZATION_CONFIG':str(config),'SLURM_JOB_ID':'synthetic-fixture'}
        result=subprocess.run([sys.executable,'-B',str(case/'analysis_worker.py'),str(prefix),str(full)],
                              cwd=case,env=env,capture_output=True,text=True)
        analysis=json.loads((case/'analysis/analysis.json').read_text())
        self.assertEqual(result.returncode,0,(result.stdout,result.stderr,analysis))
        self.assertEqual(analysis['source_realization'],contract(spec))
        self.assertEqual([phase['exit_code'] for phase in analysis['phases']],[0,0,0])
        binding=analysis['realization_circuit_binding']
        self.assertEqual(binding['raw65p_sha256'],sha256_file(full/'native_run/network65p.dat'))
        self.assertEqual(binding['normalized_input_sha256'],sha256_file(case/'analysis/network65p_assem550_divnorm.dat'))
        requested=SourceRealization(spec.circuit_id,'noise-01','retrieval','development',11,1001,2001)
        qualified=retrieval_runner.qualify_input(full,case/'analysis/formation_replay.json',
            case/'analysis/network65p_assem550_divnorm.dat',case/'analysis/normalization.json',
            case/'analysis/normalization_replay.json',coupled_extension=True,source_realization=requested)
        self.assertEqual(qualified['circuit_identity']['circuit_seed'],11)
        altered=json.loads((prefix/'native_report.json').read_text())
        altered['source_realization']=contract(replace(spec,circuit_seed=12,uniform_seed=12))
        (prefix/'native_report.json').write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError,'BUILD_AND_REQUEST_DIFFER'):
            formation_replay.compare(prefix,full,self.root/'mismatched-prefix.json',
                workdir=self.root,coupled_extension=True,source_realization_config=config)

    def test_explicit_noise_build_and_native_consumers_bind_config_before_input_access(self):
        from .ca3_realization import SourceRealization, bind_runtime, contract, verify_retrieval_program

        spec = SourceRealization('coupled-development', 'noise-01', 'retrieval', 'development', 10, 1001, 2001)
        config = self.root/'realization.yaml'
        config.write_text(json.dumps({**self.configuration, 'ca3_source_realization': spec.record()}))
        observer = dict(enabled_by=retrieval_runner.ENVIRONMENT_VARIABLE,
                        framework_revision=records.SOURCE_REVISION,
                        original_sha256=retrieval_runner.PINNED_FILES)
        with patch.object(builder, 'apply_overlay', return_value=deepcopy(observer)), \
             patch.object(memory_overlay, 'apply_overlay', return_value={'fixture': True}):
            result, work = self.build_fixture('retrieval', coupled=True,
                conductance_memory=True, source_realization_config=config)
        self.assertEqual(result['source_realization'], contract(spec))
        self.assertEqual(result['source_realization_config']['sha256'], sha256_file(config))
        selected = builder.select_project('retrieval')
        main = (work/'carlsim/projects'/selected.local_name/selected.main).read_bytes()
        rows = [row for row in result['source_file_byte_checks'] if row['path'] == selected.main]
        self.assertEqual(rows[0]['pre_realization_staged_sha256'],
                         'cf88ebeff742ce83893a8ced810bd0aca1bda5cfc058c125b88251b06ab51bfb')
        self.assertEqual(rows[0]['staged_sha256'], hashlib.sha256(main).hexdigest())
        self.assertTrue(rows[0]['source_realization_transformation_verified'])
        self.assertEqual(result['status'], 'CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING')
        self.assertFalse(result['source_realization_native_execution_qualified'])
        runtime = self.runtime_fixture(result, work, 'noise-runtime')
        args = [runtime] + [self.root/('unread-'+str(i)) for i in range(5)]
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
             patch.object(retrieval_runner, 'qualify_input') as qualify, \
             patch.object(retrieval_runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_CONFIGURATION_REQUIRED'):
                retrieval_runner.execute(*args, coupled_extension=True, native_observation=True)
            qualify.assert_not_called()
            launch.assert_not_called()
        (runtime/'realization_source.cpp').write_bytes(main)
        self.assertEqual(bind_runtime(runtime, result, config), spec)
        proof = verify_retrieval_program(main, spec)
        self.assertEqual(proof['effective_main_sha256'], result['overlay_source_sha256'])
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
             patch.object(retrieval_runner, 'qualify_input', side_effect=RuntimeError('INPUT_QUALIFICATION_REACHED')) as qualify, \
             patch.object(retrieval_runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'INPUT_QUALIFICATION_REACHED'):
                retrieval_runner.execute(*args, coupled_extension=True, native_observation=True,
                                         source_realization_config=config)
            qualify.assert_called_once()
            launch.assert_not_called()
        (runtime/'native_report.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_CONFIGURATION_REQUIRED'):
            records.validate_retrieval(runtime, runtime/'unused', coupled_extension=True)
        with self.assertRaisesRegex(ValueError, 'COMPLETED_PINNED_RETRIEVAL_REQUIRED'):
            records.validate_retrieval(runtime, runtime/'unused', coupled_extension=True,
                                       source_realization_config=config)
        changed = deepcopy(self.configuration)
        changed['ca3_source_realization'] = {**spec.record(), 'engine_seed': 2002}
        mismatch = self.root/'mismatch.yaml'
        mismatch.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'BUILD_AND_REQUEST_DIFFER'):
            bind_runtime(runtime, result, mismatch)
        with self.assertRaisesRegex(ValueError, 'PINNED_SOURCE_PROGRAM_REQUIRED'):
            verify_retrieval_program(main+b'\n', spec)
        self.assertFalse((runtime/'unused').exists())
        self.assert_originals_preserved()

    def test_unobserved_noise_build_preserves_source_for_observer_equivalence(self):
        from .ca3_realization import SourceRealization, bind_runtime

        spec = SourceRealization('coupled-development', 'noise-01', 'retrieval', 'development', 10, 1001, 2001)
        config = self.root/'realization.yaml'
        config.write_text(json.dumps({**self.configuration, 'ca3_source_realization': spec.record()}))
        with patch.object(builder, 'apply_overlay') as observe, \
             patch.object(memory_overlay, 'apply_overlay', return_value={'fixture': True}):
            result, work = self.build_fixture('retrieval', coupled=True, conductance_memory=True,
                source_realization_config=config, native_observation=False)
            observe.assert_not_called()
        self.assertNotIn('native_observation_overlay', result)
        runtime = self.runtime_fixture(result, work, 'unobserved-noise-runtime')
        selected = builder.select_project('retrieval')
        shutil.copy2(work/'carlsim/projects'/selected.local_name/selected.main, runtime/'realization_source.cpp')
        self.assertEqual(bind_runtime(runtime, result, config), spec)
        args = [runtime] + [self.root/('unread-'+str(i)) for i in range(5)]
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
             patch.object(retrieval_runner, 'qualify_input', side_effect=RuntimeError('INPUT_QUALIFICATION_REACHED')), \
             patch.object(retrieval_runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'INPUT_QUALIFICATION_REACHED'):
                retrieval_runner.execute(*args, coupled_extension=True, source_realization_config=config)
            launch.assert_not_called()
        self.assert_originals_preserved()

    def test_default_builder_import_preserves_bundles_without_optional_memory_helper(self):
        code = """import importlib.abc, sys
class NoOptionalMemory(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(('.ca3_conductance_memory', '.ca3_cue_omission', '.ca3_neuron_substeps', '.ca3_realization')):
            raise ModuleNotFoundError('optional helper absent from frozen bundle')
sys.meta_path.insert(0, NoOptionalMemory())
from population_dynamics.ca3_reference import execute
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH":str(Path(__file__).resolve().parents[1])})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_neuron_substeps_bind_only_the_integration_literal_and_require_explicit_consumers(self):
        from . import ca3_neuron_substeps as refinement
        from .ca3_retrieval_readouts import _source_proof

        self.refinement_framework_fixture = True
        observer = dict(enabled_by=retrieval_runner.ENVIRONMENT_VARIABLE,
                        framework_revision=records.SOURCE_REVISION,
                        original_sha256=retrieval_runner.PINNED_FILES)
        for count in (5, 10, 20, 40):
            with self.subTest(count=count), patch.object(builder, 'apply_overlay', return_value=deepcopy(observer)), \
                 patch.object(memory_overlay, 'apply_overlay', return_value={'fixture': 'memory overlay independently tested'}):
                result, work = self.build_fixture('retrieval', coupled=True,
                                                 conductance_memory=True, neuron_substeps=count)
                selected = builder.select_project('retrieval')
                main = (work/'carlsim/projects'/selected.local_name/selected.main).read_bytes()
                self.assertEqual(main.count(('setIntegrationMethod(RUNGE_KUTTA4, ' + str(count) + ')').encode()), 1)
                restored = main.replace(('setIntegrationMethod(RUNGE_KUTTA4, ' + str(count) + ')').encode(),
                                        b'setIntegrationMethod(RUNGE_KUTTA4, 5)', 1)
                self.assertEqual(hashlib.sha256(restored).hexdigest(), refinement.MAIN_SHA256[5])
                refinement.validate_report(result, count=count)
                records.validate_coupled_identity(result, neuron_substeps=count)
                _source_proof(result, coupled_extension=True, neuron_substeps=count)
                with self.assertRaisesRegex(ValueError, 'EXPLICIT_MATCHING_SUBSTEPS'):
                    records.validate_coupled_identity(result)
                with self.assertRaisesRegex(ValueError, 'EXPLICIT_MATCHING_SUBSTEPS'):
                    _source_proof(result, coupled_extension=True)
                changed = deepcopy(result)
                changed['numerical_refinement']['homeostasis_enabled'] = True
                with self.assertRaisesRegex(ValueError, 'BUILD_OR_REQUEST_DIFFERS'):
                    refinement.validate_report(changed, count=count)
                self.assertFalse(result['numerical_refinement']['formation_refined'])
                self.assertFalse(result['scientific_fidelity_qualified'])
        self.assert_originals_preserved()

    def test_neuron_substeps_reject_incompatible_build_requests_before_staging(self):
        valid = dict(scope='retrieval', build_only=True, local_retrieval_input=True,
                     native_observation=True, conductance_memory=True, coupled_extension_config=self.campaign)
        with patch.object(builder, 'stage') as stage:
            for count in (True, 5.0, '10', 0, 6, 80, 100):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'SUBSTEPS_MUST_BE'):
                    builder.execute(self.root/'unused', self.root/'uncreated', '89', neuron_substeps=count, **valid)
            for key, value in [('scope', 'prefix'), ('build_only', False), ('native_observation', False),
                               ('local_retrieval_input', False), ('conductance_memory', False),
                               ('coupled_extension_config', None), ('cue_omission', True)]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'REFINEMENT_REQUIRES'):
                    builder.execute(self.root/'unused', self.root/'uncreated', '89',
                                    neuron_substeps=10, **{**valid, key: value})
            stage.assert_not_called()
        self.assertFalse((self.root/'uncreated').exists())

    def test_neuron_substeps_runner_and_decoder_bind_request_before_input_access(self):
        self.refinement_framework_fixture = True
        observer = dict(enabled_by=retrieval_runner.ENVIRONMENT_VARIABLE,
                        framework_revision=records.SOURCE_REVISION,
                        original_sha256=retrieval_runner.PINNED_FILES)
        for requested in (10, 40):
            with self.subTest(requested=requested):
                with patch.object(builder, 'apply_overlay', return_value=deepcopy(observer)), \
                     patch.object(memory_overlay, 'apply_overlay', return_value={'fixture': True}):
                    result, work = self.build_fixture('retrieval', coupled=True,
                                                     conductance_memory=True, neuron_substeps=requested)
                runtime = self.runtime_fixture(result, work, 'refinement-runtime-'+str(requested))
                args = [runtime] + [self.root/('unread-' + str(i)) for i in range(5)]
                with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
                     patch.object(retrieval_runner, 'qualify_input', side_effect=RuntimeError('INPUT_QUALIFICATION_REACHED')) as qualify, \
                     patch.object(retrieval_runner.subprocess, 'Popen') as launch:
                    for count, error in [(None, 'EXPLICIT_MATCHING_SUBSTEPS'), (20, 'BUILD_OR_REQUEST_DIFFERS')]:
                        with self.subTest(count=count), self.assertRaisesRegex(ValueError, error):
                            retrieval_runner.execute(*args, coupled_extension=True, native_observation=True, neuron_substeps=count)
                    qualify.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, 'INPUT_QUALIFICATION_REACHED'):
                        retrieval_runner.execute(*args, coupled_extension=True, native_observation=True, neuron_substeps=requested)
                    launch.assert_not_called()
                (runtime/'native_report.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, 'EXPLICIT_MATCHING_SUBSTEPS'):
                    records.validate_retrieval(runtime, runtime/'uncreated-output', coupled_extension=True)
                self.assertFalse((runtime/'uncreated-output').exists())

    def test_neuron_substeps_genuine_legacy20_report_and_exact_legacy_scope(self):
        from . import ca3_neuron_substeps as refinement

        path = Path(os.environ.get('NEURONSIM_CA3_LEGACY20_BUILD_REPORT', str(
            Path(__file__).resolve().parents[2]/'Output/neural_population_dynamics/multicircuit/'
            'ca3_source_reproduction/coupled_extension/retrieval_591153/build_report.json')))
        if not path.is_file():
            self.skipTest('The immutable20-substep build receipt is not available')
        before = path.read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),
                         '83bd8f2974f2ea3446c9e5658265bb8b005fcec6f9f2c2e4a1a711b647ef387c')
        genuine = json.loads(before)
        self.assertEqual(genuine['numerical_refinement_implementation_sha256'],
                         refinement.LEGACY_IMPLEMENTATION_SHA256)
        refinement.validate_report(genuine, count=20)
        records.validate_coupled_identity(genuine, neuron_substeps=20)
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_MATCHING_SUBSTEPS'):
            refinement.validate_report(genuine, count=None)
        for count in (5, 10, 20, 40):
            # Only the20case above is native evidence. These copies test the
            # helper-hash scope without invoking a binary or changing that receipt.
            candidate = deepcopy(genuine)
            candidate.update(integration='RUNGE_KUTTA4_'+str(count),
                             numerical_refinement=refinement.contract(count),
                             overlay_source_sha256=refinement.MAIN_SHA256[count])
            if count == 40:
                with self.assertRaisesRegex(ValueError, 'BUILD_OR_REQUEST_DIFFERS'):
                    refinement.validate_report(candidate, count=40)
                candidate['numerical_refinement_implementation_sha256'] = sha256_file(Path(refinement.__file__))
            refinement.validate_report(candidate, count=count)
            candidate['numerical_refinement_implementation_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'BUILD_OR_REQUEST_DIFFERS'):
                refinement.validate_report(candidate, count=count)
        self.assertEqual(path.read_bytes(), before)

    def test_neuron_substeps_clis_accept40_and_reject80_before_execution(self):
        from . import carlsim_retrieval_replay as replay

        stub = dict(status='CLI_FIXTURE', retrieval_executed=False, scientific_fidelity_qualified=False)
        cases = [
            (builder, 'execute', ['--source-root','unused','--workdir','unused','--cuda-arch','89']),
            (retrieval_runner, 'execute', [argument for name in (
                'runtime','formation','formation-replay','normalized','normalization','normalization-replay')
                for argument in ('--'+name,'unused')]),
            (records, 'validate_retrieval', ['--runtime','unused','--output','unused','--scope','retrieval']),
            (replay, 'compare', ['--original','unused-a','--repeat','unused-b','--output','unused-c']),
        ]
        for module, function, arguments in cases:
            with self.subTest(module=module.__name__):
                with patch.object(module,function,return_value=stub) as execute, \
                     patch.object(sys,'argv',[module.__name__,*arguments,'--neuron-substeps','40']), \
                     redirect_stdout(io.StringIO()):
                    module.main()
                self.assertEqual(execute.call_args.kwargs['neuron_substeps'],40)
                with patch.object(module,function) as execute, \
                     patch.object(sys,'argv',[module.__name__,*arguments,'--neuron-substeps','80']), \
                     redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                    module.main()
                self.assertEqual(failure.exception.code,2)
                execute.assert_not_called()

    def observed_omission_fixture(self, coupled):
        metadata = dict(enabled_by=retrieval_runner.ENVIRONMENT_VARIABLE,
                        framework_revision=records.SOURCE_REVISION,
                        original_sha256=retrieval_runner.PINNED_FILES)
        with patch.object(builder, 'apply_overlay', return_value=metadata):
            return self.build_fixture('retrieval', coupled=coupled, cue_omission=True)

    def test_omission_build_retains_complete_source_except_one_assignment(self):
        for coupled in (False, True):
            result, work = self.observed_omission_fixture(coupled)
            multiplier = 2 if coupled else 1
            omission.validate_report(result, population_multiplier=multiplier)
            selected = builder.select_project('retrieval')
            path = work/'carlsim/projects'/selected.local_name/selected.main
            transformed = path.read_bytes()
            self.assertEqual(transformed.count(omission.OMITTED_ASSIGNMENT), 1)
            restored = transformed.replace(omission.OMITTED_ASSIGNMENT, omission.SOURCE_ASSIGNMENT, 1)
            self.assertEqual(hashlib.sha256(restored).hexdigest(), omission.SOURCE_MAIN_SHA256[multiplier])
            self.assertFalse(result['retrieval_input_contract']['source_path_binding_only'])
            self.assertFalse(result['author_program_semantics_preserved_before_stop'])
            self.assertFalse(result['native_run_executed'])
        self.assert_originals_preserved()

    def test_omission_execution_requires_matching_explicit_condition_before_input_access(self):
        result, work = self.observed_omission_fixture(True)
        runtime = self.runtime_fixture(result, work, 'omission-runtime')
        args = [runtime]+[self.root/('unread-'+str(i)) for i in range(5)]
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fixture', 'CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
             patch.object(retrieval_runner, 'qualify_input', side_effect=RuntimeError('INPUT_QUALIFICATION_REACHED')) as qualify, \
             patch.object(retrieval_runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'CONDITION_FLAGS_DIFFER'):
                retrieval_runner.execute(*args, coupled_extension=True, native_observation=True)
            qualify.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'OMISSION_REQUIRES_NATIVE_OBSERVATION'):
                retrieval_runner.execute(*args, coupled_extension=True, cue_omission=True)
            qualify.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'INPUT_QUALIFICATION_REACHED'):
                retrieval_runner.execute(*args, coupled_extension=True, native_observation=True, cue_omission=True)
            self.assertEqual(qualify.call_count, 1)
            launch.assert_not_called()
        self.assertFalse((runtime/'native_run').exists())

    def test_omission_cannot_be_decoded_or_read_out_as_original_source(self):
        from .ca3_retrieval_readouts import _source_proof

        result, work = self.observed_omission_fixture(True)
        runtime = self.runtime_fixture(result, work, 'omission-decoder')
        (runtime/'native_report.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'CONDITION_FLAGS_DIFFER'):
            records.validate_retrieval(runtime, runtime/'rejected-observations', coupled_extension=True)
        with self.assertRaisesRegex(ValueError, 'SOURCE_CONDITION'):
            _source_proof(result, coupled_extension=True)

    def test_omission_build_rejects_unobserved_or_nonretrieval_before_staging(self):
        with patch.object(builder, 'stage') as stage:
            for options in [dict(scope='prefix', local_retrieval_input=True, native_observation=True),
                            dict(scope='retrieval', local_retrieval_input=True, native_observation=False),
                            dict(scope='retrieval', local_retrieval_input=False, native_observation=True)]:
                with self.assertRaisesRegex(ValueError, 'OMISSION_REQUIRES_OBSERVED_LOCAL_RETRIEVAL_BUILD'):
                    builder.execute(self.root/'unused', self.root/'uncreated', '89', cue_omission=True, **options)
            stage.assert_not_called()
        self.assertFalse((self.root/'uncreated').exists())

    def test_conductance_memory_overlay_is_explicit_and_retained_in_build_report(self):
        metadata = {"overlay_id": "ca3_conductance_memory", "native_qualification": False}
        with patch.object(memory_overlay, "apply_overlay", return_value=metadata) as overlay:
            result, work = self.build_fixture("prefix", coupled=True, conductance_memory=True)
        self.assertEqual(result["conductance_memory_overlay"], metadata)
        overlay.assert_called_once_with(self.root / "declared-staging-inputs/carlsim_ca3",
                                        work / "carlsim", preceding_overlay=None)
        self.assertEqual(result["conductance_memory_implementation_sha256"],
                         sha256_file(Path(builder.__file__).with_name("ca3_conductance_memory.py")))
        self.assertFalse(result["source_code_semantics_preserved_before_stop"])
        self.assertFalse(result["native_run_executed"])
        self.assert_originals_preserved()

    def test_native_memory_builds_preserve_author_program_in_all_scopes(self):
        metadata = {"overlay_id": "ca3_conductance_memory", "native_qualification": False}
        for scope in builder.SCOPES:
            with self.subTest(scope=scope), patch.object(memory_overlay, "apply_overlay", return_value=metadata):
                result, work = self.build_fixture(scope, coupled=False, conductance_memory=True)
                selected = builder.select_project(scope)
                self.assertTrue(result["author_program_semantics_preserved_before_stop"])
                self.assertFalse(result["source_code_semantics_preserved_before_stop"])
                self.assertEqual(result["core_neurons"], 84053)
                self.assertEqual(result["pattern_integer_count"], 1100)
                self.assertEqual(result["pattern_sha256"], sha256_file(self.author / selected.patterns))
                self.assertEqual(result["expected_duration_ms"], {
                    "prefix": 4450, "full-formation": 70000, "retrieval": 1001}[scope])
                source = self.original[selected.project + "/" + selected.main]
                effective = work / "carlsim/projects" / selected.local_name / selected.main
                if scope == "prefix":
                    source = builder.formation_program(source.decode().replace("\r\n", "\n"), scope).encode()
                self.assertEqual(effective.read_bytes(), source)
                self.assertEqual(result["overlay_source_sha256"], sha256_file(effective))
                self.assertEqual(result["conductance_memory_overlay"], metadata)
                self.assertEqual(result["complete_author_formation_program_unchanged"], scope == "full-formation")
                self.assertEqual(result["complete_author_retrieval_program_unchanged"], scope == "retrieval")
                self.assertFalse(result["full_scale_qualified"])
                self.assertFalse(result["scientific_fidelity_qualified"])
        self.assert_originals_preserved()

    def test_native_prefix_rejects_changes_beyond_declared_stop(self):
        prepare = builder.prepare_formation_program

        def changed_program(path, scope):
            prepare(path, scope)
            path.write_text(path.read_text() + "\nvoid undeclared_source_change() {}\n")

        with patch.object(builder, "prepare_formation_program", side_effect=changed_program):
            with self.assertRaisesRegex(ValueError, "UNDECLARED_NATIVE_PREFIX_SOURCE_CHANGE"):
                self.build_fixture("prefix", coupled=False)
        self.assert_originals_preserved()

    def runtime_fixture(self, result, work, name):
        runtime = self.root / name
        binary_name = "ca3_retrieval" if result["scope"].endswith("retrieval") else "ca3_formation"
        (runtime / "bin").mkdir(parents=True)
        shutil.copy2(work / result["binary_relative_path"], runtime / "bin" / binary_name)
        shutil.copy2(work / result["effective_pattern_relative_path"], runtime / "trainPat.csv")
        (runtime / "build_report.json").write_text(json.dumps(result))
        names = ["bin/" + binary_name, "trainPat.csv", "build_report.json"]
        (runtime / "manifest.json").write_text(json.dumps({
            "files": [{"path": relative, "sha256": sha256_file(runtime / relative)} for relative in names],
            "reference_checkpoint_bytes": 1,
        }))
        return runtime

    def test_real_project_preparation_preserves_sources_and_effective_pattern(self):
        for scope, role in (("full-formation", "formation"), ("retrieval", "retrieval")):
            selected, project, local, patterns = self.copied_project(scope, role)
            metadata, expected, effective = builder.prepare_coupled_project(
                project, local, patterns, selected, multiplier=2, role=role)
            with self.subTest(role=role):
                counts, probabilities = native_parameters((local / "generateCONFIGStateSTP.h").read_text())
                self.assertEqual(sum(counts.values()), 168106)
                self.assertEqual(len(probabilities), 51)
                self.assertEqual(metadata["derived_assembly_size"], 550)
                self.assertEqual(metadata["derived_cue_size"], 550 if role == "formation" else 276)
                self.assertEqual(effective, local / "trainPat.csv")
                original_indices = list(map(int, self.original[selected.patterns].split()))
                self.assertEqual(list(map(int, effective.read_text().split())),
                                 [child for index in original_indices for child in (2 * index, 2 * index + 1)])
                self.assertEqual(metadata["source_pattern_sha256"], sha256_file(patterns))
                self.assertEqual(metadata["derived_pattern_sha256"], sha256_file(effective))
                for name, content in expected.items():
                    self.assertEqual((local / name).read_bytes(), content)
                self.assertEqual((local / "generateSETUPStateSTP.h").read_bytes(),
                                 self.original[selected.project + "/generateSETUPStateSTP.h"])
        self.assert_originals_preserved()

    def test_derived_retrieval_requires_local_separate_checkpoint_binding(self):
        selected, project, local, patterns = self.copied_project("retrieval", "derived-binding")
        builder.prepare_coupled_project(project, local, patterns, selected, multiplier=2, role="retrieval")
        main, header = local / selected.main, local / "generateCONFIGStateSTP.h"
        before = main.read_bytes()
        with self.assertRaisesRegex(ValueError, "COUPLED_RETRIEVAL_REQUIRES_LOCAL_INPUT"):
            builder.prepare_retrieval_program(main, header, population_multiplier=2)
        self.assertEqual(main.read_bytes(), before)
        contract = builder.prepare_retrieval_program(main, header, local_input=True, population_multiplier=2)
        self.assertEqual(contract["required_checkpoint"], "network65p_assem550_divnorm.dat")
        self.assertEqual(contract["effective_input_binding"], contract["required_checkpoint"])
        self.assertTrue(contract["fresh_coupled_checkpoint_required"])
        self.assertFalse(contract["source_path_binding_only"])
        self.assertFalse(contract["normalized_input_qualified"])
        self.assertEqual(main.read_bytes(), before.replace(
            ('"' + contract["original_source_input_binding"] + '"').encode(),
            b'"network65p_assem550_divnorm.dat"', 1))
        selected, _, reference, _ = self.copied_project("retrieval", "reference-binding")
        original_contract = builder.prepare_retrieval_program(
            reference / selected.main, reference / "generateCONFIGStateSTP.h", local_input=True)
        self.assertEqual(original_contract["required_checkpoint"], "network65p_assem275_divnorm.dat")
        self.assertTrue(original_contract["source_path_binding_only"])
        self.assertFalse(original_contract["fresh_coupled_checkpoint_required"])
        self.assert_originals_preserved()

    def test_build_only_records_distinct_scope_and_actual_derived_pattern(self):
        for scope in builder.SCOPES:
            result, work = self.build_fixture(scope, coupled=True)
            selected = builder.select_project(scope)
            with self.subTest(scope=scope):
                self.assertEqual(result["status"], "COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED")
                self.assertEqual(result["scope"], "coupled_extension_" + ("formation" if scope == "full-formation" else scope))
                self.assertEqual(result["core_neurons"], 168106)
                self.assertEqual(result["pattern_integer_count"], 2200)
                effective = work / result["effective_pattern_relative_path"]
                self.assertEqual(effective, work / "carlsim/projects" / selected.local_name / "trainPat.csv")
                self.assertEqual(result["pattern_sha256"], sha256_file(effective))
                self.assertEqual(result["coupled_extension"]["derived_pattern_sha256"], result["pattern_sha256"])
                self.assertEqual(result["coupled_extension_config"]["sha256"], sha256_file(self.campaign))
                self.assertEqual(result["coupled_extension_config"]["resolved"], {"population_multiplier": 2})
                self.assertEqual((work / "author" / selected.patterns).read_bytes(), self.original[selected.patterns])
                self.assertFalse(result["source_code_semantics_preserved_before_stop"])
                self.assertFalse(result["author_program_semantics_preserved_before_stop"])
                self.assertFalse(result["complete_author_formation_program_unchanged"])
                self.assertFalse(result["complete_author_retrieval_program_unchanged"])
                self.assertFalse(result["full_scale_qualified"])
                self.assertFalse(result["scientific_fidelity_qualified"])
                self.assertTrue(all(row["coupled_extension_transformation_verified"] for row in result["source_file_byte_checks"]))
                self.assertEqual(next(row for row in result["source_file_byte_checks"] if row["path"] == selected.main)["staged_sha256"],
                                 result["overlay_source_sha256"])
                if scope == "retrieval":
                    self.assertEqual(result["retrieval_input_contract"]["required_checkpoint"], "network65p_assem550_divnorm.dat")
        self.assert_originals_preserved()

    def test_no_extension_argument_preserves_source_identity_and_pattern(self):
        for scope in ("full-formation", "retrieval"):
            result, work = self.build_fixture(scope, coupled=False)
            selected = builder.select_project(scope)
            with self.subTest(scope=scope):
                self.assertEqual(result["scope"], "native_source_formation" if scope == "full-formation" else "native_source_retrieval")
                self.assertEqual(result["status"], "NATIVE_SOURCE_BUILD_PASSED_GPU_RUN_PENDING" if scope == "full-formation"
                                 else "NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING")
                self.assertEqual(result["core_neurons"], 84053)
                self.assertEqual(result["pattern_integer_count"], 1100)
                self.assertNotIn("coupled_extension", result)
                self.assertNotIn("coupled_extension_config", result)
                self.assertEqual(work / result["effective_pattern_relative_path"], work / "author" / selected.patterns)
                self.assertEqual(result["pattern_sha256"], sha256_file(self.author / selected.patterns))
                self.assertTrue(result["source_code_semantics_preserved_before_stop"])
                self.assertTrue(all(row["source_bytes_preserved"] for row in result["source_file_byte_checks"]))
                if scope == "retrieval":
                    self.assertEqual(result["retrieval_input_contract"]["required_checkpoint"], "network65p_assem275_divnorm.dat")
        self.assert_originals_preserved()

    def test_invalid_extension_configuration_and_absolute_retrieval_fail_before_staging(self):
        invalid = [None, {}, {"population_multiplier": 1}, {"population_multiplier": True},
                   {"population_multiplier": 2.0}, {"population_multiplier": 3},
                   {"population_multiplier": 2, "unapproved_option": 1}]
        for index, options in enumerate(invalid):
            config = deepcopy(self.configuration)
            config["ca3_coupled_extension"] = options
            path, work = self.root / f"invalid-{index}.yaml", self.root / f"uncreated-{index}"
            path.write_text(json.dumps(config))
            with patch.object(builder, "stage") as stage, patch.object(builder.subprocess, "run") as run:
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "COUPLED_EXTENSION_CONFIG_REQUIRED"):
                    builder.execute(self.root / "unused", work, "89", build_only=True,
                                    coupled_extension_config=path)
                stage.assert_not_called()
                run.assert_not_called()
            self.assertFalse(work.exists())
        work = self.root / "uncreated-absolute-retrieval"
        with patch.object(builder, "stage") as stage, patch.object(builder.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "COUPLED_RETRIEVAL_REQUIRES_LOCAL_INPUT"):
                builder.execute(self.root / "unused", work, "89", build_only=True, scope="retrieval",
                                coupled_extension_config=self.campaign)
            stage.assert_not_called()
            run.assert_not_called()
        self.assertFalse(work.exists())

    def test_native_execution_admission_rejects_derived_build_before_old_input_or_launch(self):
        formation, work = self.build_fixture("full-formation", coupled=True)
        runtime = self.runtime_fixture(formation, work, "native-formation-rejection")
        with patch.dict(os.environ, {"SLURM_JOB_ID": "software-fixture", "CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
             patch.object(formation_runner, "probe") as probe, patch.object(formation_runner.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "UNCHANGED_FULL_FORMATION_BUILD_REQUIRED"):
                formation_runner.execute(runtime)
            probe.assert_not_called()
            launch.assert_not_called()
        self.assertFalse((runtime / "native_run").exists())
        retrieval, work = self.build_fixture("retrieval", coupled=True)
        runtime = self.runtime_fixture(retrieval, work, "native-retrieval-rejection")
        old_input = self.root / "unread-old-84053-reference-checkpoint.dat"
        with patch.dict(os.environ, {"SLURM_JOB_ID": "software-fixture", "CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
             patch.object(retrieval_runner, "qualify_input") as consume, \
             patch.object(retrieval_runner, "probe") as probe, patch.object(retrieval_runner.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "PATH_ONLY_RETRIEVAL_BUILD_REQUIRED"):
                retrieval_runner.execute(runtime, self.root / "old-formation", self.root / "old-replay.json",
                                         old_input, self.root / "old-normalization.json", self.root / "old-normalization-replay.json")
            consume.assert_not_called()
            probe.assert_not_called()
            launch.assert_not_called()
        self.assertFalse((runtime / "native_run").exists())
        self.assertFalse(old_input.exists())

    def test_existing_checkpoint_validator_rejects_reference_header_for_derived_count(self):
        # Metadata-only CARLsim-format fixture; no synaptic payload or simulator claim.
        counts, _ = native_parameters(self.original[builder.PROJECT + "/generateCONFIGStateSTP.h"].decode())
        checkpoint = self.root / "network5p.dat"
        payload = records.NETWORK_HEADER.pack(records.NETWORK_SIGNATURE, 0.3, 4.45, 0.0, 84053, len(counts))
        start = 0
        for name, count in counts.items():
            payload += records.GROUP_HEADER.pack(start, start + count - 1, count, 1, 1, name.encode())
            start += count
        checkpoint.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "CHECKPOINT_UNIT_IDENTITY_DIFFERS"):
            builder.verify_checkpoints(self.root, "prefix", expected_units=168106)


if __name__ == "__main__":
    unittest.main()


class BackendConfigurationBinding(unittest.TestCase):
    """Exercise the runtime consumer without compiling or loading a backend."""
    def setUp(self):
        import ast
        import yaml

        self.temporary = tempfile.TemporaryDirectory(prefix='ca3-backend-config-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        repository = Path(__file__).resolve().parents[2]
        launcher = repository/'Output/neural_population_dynamics/operations/ca3_coupled_runtime.sbatch'
        tree=ast.parse(launcher.read_text().split("<<'PYTHON'\n",1)[1].rsplit('\nPYTHON',1)[0])
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                      and node.name=='backend_configuration_binding')
        namespace={'hashlib':hashlib,'yaml':yaml}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<runtime-configuration-binding>','exec'),namespace)
        self.check=namespace['backend_configuration_binding']
        self.current=self.root/'current';self.reference=self.root/'formation'
        self.name='Libs/configs/neural_population_dynamics.yaml'
        self.left,self.left_row=self.config(self.current, unrelated='updated model settings')
        self.right,self.right_row=self.config(self.reference, unrelated='previous model settings')
        self.manifest={self.name:self.right_row}
        self.delivery={'formation/'+self.name:dict(self.right_row)}

    def config(self, root, *, unrelated, multiplier=2, resolved=None):
        import yaml

        selected={'population_multiplier':multiplier}
        path=root/self.name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump({'ca3_coupled_extension':selected,'unrelated_model':unrelated}))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(coupled_extension_config=dict(sha256=digest,resolved=selected if resolved is None else resolved)), \
            dict(sha256=digest,bytes=path.stat().st_size)

    def call(self):
        return self.check(self.current,self.left,self.left_row['sha256'],self.reference,self.right,self.manifest,self.delivery)

    def test_unrelated_global_config_changes_retain_both_full_hash_bindings(self):
        result=self.call()
        self.assertNotEqual(result['current']['full_config_sha256'],result['reference']['full_config_sha256'])
        self.assertTrue(result['each_full_configuration_hash_verified'])
        self.assertTrue(result['selected_ca3_configuration_equal'])
        self.assertEqual(result['compared_source_configuration_fields'],['ca3_coupled_extension.population_multiplier'])

    def test_changed_actual_ca3_domain_or_type_is_rejected(self):
        for value in [3,True,2.0]:
            with self.subTest(multiplier=value):
                self.left,self.left_row=self.config(self.current,unrelated='same',multiplier=value)
                with self.assertRaisesRegex(ValueError,'SELECTED_CA3_CONFIGURATION_DIFFERS'):
                    self.call()

    def test_changed_yaml_bytes_are_rejected_before_semantic_comparison(self):
        for root in [self.current,self.reference]:
            path=root/self.name;raw=path.read_bytes();path.write_bytes(raw+b'\n# changed after binding\n')
            with self.subTest(root=root.name),self.assertRaisesRegex(ValueError,'FULL_CONFIG_BINDING_DIFFERS'):
                self.call()
            path.write_bytes(raw)

    def test_tampered_manifest_delivery_binding_is_rejected(self):
        for field,value in [('sha256','0'*64),('bytes',0)]:
            saved=dict(self.delivery['formation/'+self.name]);self.delivery['formation/'+self.name][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'CONFIG_MANIFEST_DELIVERY_DIFFERS'):
                self.call()
            self.delivery['formation/'+self.name]=saved

    def test_tampered_build_full_hash_is_rejected(self):
        self.right['coupled_extension_config']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'FULL_CONFIG_BINDING_DIFFERS'):
            self.call()

    def test_tampered_resolved_or_extra_ca3_field_is_rejected(self):
        for resolved in [{'population_multiplier':3},{'population_multiplier':2,'another_control':1}]:
            self.left['coupled_extension_config']['resolved']=resolved
            with self.subTest(resolved=resolved),self.assertRaisesRegex(ValueError,'SELECTED_CA3_CONFIGURATION_DIFFERS'):
                self.call()
