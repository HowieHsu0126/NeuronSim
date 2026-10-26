import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import ca3_reference as builder
from .provenance import sha256_file


RETRIEVAL_MAIN = '''int main() {
int randSeed = 10;
int myrandom = 3;
std::srand(myrandom);
sim.setIntegrationMethod(RUNGE_KUTTA4, 5);
fId = fopen("/source/network65p_assem275_divnorm.dat", "rb");
sim.loadSimulation(fId);
return 0;
}
'''.replace('\n','\r\n').encode()
CONFIG = ('\n'.join(f'sim.createGroup("G{i}", {i+1});' for i in range(8)) + '''
sim.connect(CA3_Pyramidal, CA3_Pyramidal, RangeDelay(1));
sim.setSTP(CA3_Pyramidal, CA3_Pyramidal, true);
sim.setESTDP(CA3_Pyramidal, CA3_Pyramidal, true, STANDARD);
''').replace('\n','\r\n').encode()
FORMATION_MAIN = b'sim.setIntegrationMethod(RUNGE_KUTTA4, 5);\r\nsim.saveSimulation("network5p.dat", true);\r\nnext_cycle();\r\n'


def sources(root):
    for source, files in (
        ('carlsim_ca3', {'CMakeLists.txt':b'cmake_minimum_required(VERSION 3.10)\n'}),
        ('ca3_formation_retrieval', {
            builder.PROJECT+'/'+builder.MAIN:FORMATION_MAIN,
            builder.RETRIEVAL_PROJECT+'/'+builder.RETRIEVAL_MAIN:RETRIEVAL_MAIN,
            **{project+'/'+name:payload for project in (builder.PROJECT,builder.RETRIEVAL_PROJECT)
               for name,payload in [('generateCONFIGStateSTP.h',CONFIG),('generateSETUPStateSTP.h',b'// source monitor\r\n')]},
            builder.PATTERNS:b' '.join(str(i).encode() for i in range(20)),
            builder.RETRIEVAL_PATTERNS:b' '.join(str(i).encode() for i in range(20)),
        })):
        folder=root/source;folder.mkdir(parents=True)
        rows=[]
        for name,payload in files.items():
            p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
            rows.append({'path':name,'sha256':sha256_file(p)})
        (folder/'source.json').write_text(json.dumps({'files':rows,'deferred_blobs':[],'revision':source+'_pinned'}))


class RetrievalBuild(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_phase_selection_preserves_formation_targets_and_rejects_bad_scopes(self):
        self.assertEqual(builder.select_project('prefix'),builder.select_project('full-formation'))
        self.assertEqual(builder.select_project('prefix').target,builder.TARGET)
        selected=builder.select_project('retrieval')
        self.assertEqual(selected.project,builder.RETRIEVAL_PROJECT)
        self.assertEqual(selected.target,builder.RETRIEVAL_TARGET)
        self.assertEqual(selected.local_name,'author_retrieval')
        with self.assertRaisesRegex(ValueError,'SCOPE'):builder.select_project('invalid')
        with self.assertRaisesRegex(ValueError,'SCOPE'):builder.verify_checkpoints(self.root,'retrieval',expected_units=5)

    def test_retrieval_inspection_preserves_exact_crlf_source_and_input_contract(self):
        main,config=self.root/'main.cpp',self.root/'config.h'
        main.write_bytes(RETRIEVAL_MAIN);config.write_bytes(CONFIG)
        result=builder.prepare_retrieval_program(main,config)
        self.assertEqual(main.read_bytes(),RETRIEVAL_MAIN)
        self.assertEqual(config.read_bytes(),CONFIG)
        self.assertEqual(result['required_checkpoint'],'network65p_assem275_divnorm.dat')
        self.assertTrue(result['source_estdp_enabled'])
        self.assertFalse(result['raw_network65p_substitution_allowed'])
        self.assertFalse(result['normalized_input_qualified'])
        self.assertFalse(result['native_retrieval_executed'])

    def test_local_binding_changes_only_the_input_path_literal(self):
        main,config=self.root/'main.cpp',self.root/'config.h'
        main.write_bytes(RETRIEVAL_MAIN);config.write_bytes(CONFIG)
        result=builder.prepare_retrieval_program(main,config,local_input=True)
        expected=RETRIEVAL_MAIN.replace(b'"/source/network65p_assem275_divnorm.dat"',b'"network65p_assem275_divnorm.dat"')
        self.assertEqual(main.read_bytes(),expected)
        self.assertEqual(config.read_bytes(),CONFIG)
        self.assertEqual(result['effective_input_binding'],result['required_checkpoint'])
        self.assertEqual(result['original_source_input_binding'],'/source/network65p_assem275_divnorm.dat')
        self.assertTrue(result['source_path_binding_only'])
        self.assertFalse(result['normalized_input_qualified'])
        with self.assertRaisesRegex(ValueError,'RETRIEVAL_ONLY'):
            builder.execute(self.root,self.root/'unused','89',build_only=True,local_retrieval_input=True)
        self.assertFalse((self.root/'unused').exists())

    def test_changed_normalized_binding_seeds_and_estdp_are_rejected(self):
        main,config=self.root/'main.cpp',self.root/'config.h'
        for old,new,error in [(b'network65p_assem275_divnorm.dat',b'network65p.dat','NORMALIZED65P'),
                              (b'randSeed = 10',b'randSeed = 11','RANDOM_STREAM'),
                              (b'myrandom = 3',b'myrandom = 4','RANDOM_STREAM')]:
            with self.subTest(error=error):
                main.write_bytes(RETRIEVAL_MAIN.replace(old,new));config.write_bytes(CONFIG)
                with self.assertRaisesRegex(ValueError,error):builder.prepare_retrieval_program(main,config)
        main.write_bytes(RETRIEVAL_MAIN)
        config.write_bytes(CONFIG.replace(b'setESTDP(CA3_Pyramidal, CA3_Pyramidal, true',b'setESTDP(CA3_Pyramidal, CA3_Pyramidal, false'))
        with self.assertRaisesRegex(ValueError,'ESTDP'):builder.prepare_retrieval_program(main,config)

    def test_retrieval_cannot_run_or_create_outputs_through_builder(self):
        output=self.root/'must_not_exist'
        with patch.object(builder.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'BUILD_ONLY_REQUIRES_QUALIFIED_NORMALIZED65P'):
                builder.execute(self.root/'missing_sources',output,'89',scope='retrieval')
            run.assert_not_called()
        self.assertFalse(output.exists())

    def test_shared_build_path_selects_target_and_retains_all_source_bytes(self):
        source=self.root/'sources';sources(source)
        for scope in builder.SCOPES:
            with self.subTest(scope=scope):
                selected=builder.select_project(scope);work=self.root/scope
                def fake_run(command,**kwargs):
                    if command[1]=='--build':
                        target=Path(command[2])/'projects'/selected.local_name/selected.target
                        target.parent.mkdir(parents=True);target.write_bytes(b'build fixture')
                with patch.dict(os.environ,{'SLURM_JOB_ID':'fixture','SLURM_CPUS_PER_TASK':'1','CUDA_VISIBLE_DEVICES':''}), \
                     patch.object(builder,'materialize_build_inputs',return_value={'fixture':True}), \
                     patch.object(builder.subprocess,'run',side_effect=fake_run) as run, \
                     patch.object(builder.subprocess,'check_output',return_value='fixture toolchain'):
                    result=builder.execute(source,work,'89',scope=scope,build_only=True)
                self.assertEqual(run.call_count,2)
                self.assertEqual(result['build_target'],selected.target)
                self.assertEqual(result['build_commands'][1][-3:],[selected.target,'--parallel','1'])
                self.assertFalse(result['native_run_executed'])
                self.assertFalse((work/'native_run').exists())
                self.assertEqual(sha256_file(work/'author'/selected.patterns),result['pattern_sha256'])
                staged=work/'carlsim/projects'/selected.local_name
                self.assertEqual((staged/'generateCONFIGStateSTP.h').read_bytes(),CONFIG)
                self.assertEqual((staged/'generateSETUPStateSTP.h').read_bytes(),b'// source monitor\r\n')
                self.assertIn('add_subdirectory(projects/'+selected.local_name+')',(work/'carlsim/CMakeLists.txt').read_text())
                if scope=='retrieval':
                    self.assertEqual(result['scope'],'native_source_retrieval')
                    self.assertEqual(result['expected_duration_ms'],1001)
                    self.assertTrue(result['complete_author_retrieval_program_unchanged'])
                    self.assertTrue(all(c['source_bytes_preserved'] for c in result['source_file_byte_checks']))
                    self.assertEqual((staged/selected.main).read_bytes(),RETRIEVAL_MAIN)
                    self.assertEqual(result['status'],'NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING')
                elif scope=='full-formation':
                    self.assertEqual((staged/selected.main).read_bytes(),FORMATION_MAIN)
                    self.assertTrue(result['complete_author_formation_program_unchanged'])
                    self.assertEqual(result['status'],'NATIVE_SOURCE_BUILD_PASSED_GPU_RUN_PENDING')
                else:
                    self.assertIn('declared source-prefix stop',(staged/selected.main).read_text())
                    self.assertEqual(result['expected_prefix_ms'],4450)

    def test_cli_retrieval_without_build_only_fails_before_staging(self):
        output=self.root/'no_native'
        command=[sys.executable,'-m','population_dynamics.ca3_reference','--source-root',str(self.root),
                 '--workdir',str(output),'--cuda-arch','89','--scope','retrieval']
        result=subprocess.run(command,capture_output=True,text=True,
                              env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])})
        self.assertNotEqual(result.returncode,0)
        self.assertIn('BUILD_ONLY_REQUIRES_QUALIFIED_NORMALIZED65P',result.stderr)
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
