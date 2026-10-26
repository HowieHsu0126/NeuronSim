import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import BadZipFile

import numpy as np

from . import carlsim_retrieval_replay as replay
from . import ca3_native_observation as observation
from . import ca3_neuron_substeps as substeps
from .carlsim_records import SOURCE_REVISION, SPIKE_HEADER, SPIKE_SIGNATURE, _sha, validate_retrieval
from .test_carlsim_records import events, network
from .test_carlsim_formation_replay import coupled_declaration, coupled_network


E_ROWS = [(520, 2), (520, 0), (520, 2), (1, 0), (1000, 1)]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def update_report(root, values, *, bind=True):
    path = root/'native_report.json'
    data = json.loads(path.read_text()); data.update(values); write(path, data)
    if bind:
        receipt = root/'observations/recording.json'
        row = json.loads(receipt.read_text()); row['native_report_sha256'] = _sha(path)
        write(receipt, row)


def refresh_manifest(root):
    names = ('build_report.json', 'bin/ca3_retrieval', 'trainPat.csv', 'lib/libcarlsim.so')
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['files'] = [{'path':name, 'sha256':_sha(root/name)} for name in names]
    write(root/'manifest.json', manifest)
    update_report(root, {'runtime_manifest_sha256':_sha(root/'manifest.json')})


def fixture(root, *, repeat=False, e_rows=None, coupled_extension=False):
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    input_name = 'network65p_assem550_divnorm.dat' if coupled_extension else replay.INPUT_NAME
    run = root/'native_run'
    (run/'results').mkdir(parents=True)
    (root/'bin').mkdir(); (root/'lib').mkdir()
    (root/'bin/ca3_retrieval').write_bytes(b'fixture binary bytes; never executed')
    (root/'lib/libcarlsim.so').write_bytes(b'fixture library')
    pattern = ' '.join(str(i) for i in list(range(550))*4)+'\n' if coupled_extension else '0 1 2\n'
    (root/'trainPat.csv').write_text(pattern)
    shutil.copy2(root/'trainPat.csv', run/'trainPat.csv')
    (coupled_network if coupled_extension else network)(root/'normalized.dat', seconds=45.85)
    with (root/'normalized.dat').open('ab') as stream: stream.write(b'fixed normalized payload')
    (run/input_name).symlink_to(root/'normalized.dat')
    if coupled_extension:
        for name,count in coupled_declaration()['derived_populations'].items():
            rows = [(0,0),(1000,count-1)]
            if name == 'CA3_Pyramidal':
                rows = [*E_ROWS[:-1],(1000,count-1)] if e_rows is None else e_rows
            events(run/'results'/('spk_'+name+'.dat'),count,rows)
    else:
        events(run/'results/spk_I.dat', 2, [(0, 0), (1000, 1)])
        events(run/'results/spk_E.dat', 3, E_ROWS if e_rows is None else e_rows)
    build = dict(status='NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING',
        scope='native_source_retrieval', framework_revision=SOURCE_REVISION, author_revision=replay.AUTHOR_REVISION,
        binary_sha256=_sha(root/'bin/ca3_retrieval'), core_neurons=5, populations={'I':2, 'E':3},
        integration='RUNGE_KUTTA4_5', delay_ms=1,
        source_rng='CARLsim_seed10_and_author_default_host_streams_not_independent_stream_qualified',
        pattern_sha256=_sha(root/'trainPat.csv'), pattern_integer_count=3,
        original_author_main_sha256='a'*64, overlay_source_sha256='b'*64,
        source_file_byte_checks=[dict(path='main.cpp',source_sha256='a'*64,staged_sha256='b'*64,
                                     source_bytes_preserved=False,source_path_binding_only=True)],
        retrieval_input_contract=dict(effective_input_binding=replay.INPUT_NAME, source_path_binding_only=True,
            source_carlsim_seed=10, source_host_srand_seed=3, source_estdp_enabled=True),
        retrieval_program_changed_only_in_input_path=True)
    if coupled_extension:
        extension = coupled_declaration()
        extension.update(source_role='retrieval',source_cue_size=138,derived_cue_size=276,
                         derived_pattern_sha256=build['pattern_sha256'])
        build.update(status='COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED',
            scope='coupled_extension_retrieval',core_neurons=168106,populations=extension['derived_populations'],
            source_code_semantics_preserved_before_stop=False,coupled_extension=extension,
            coupled_extension_config=dict(resolved={'population_multiplier':2},sha256='8'*64),
            pattern_integer_count=2200,retrieval_program_changed_only_in_input_path=False,
            retrieval_input_contract=dict(effective_input_binding=input_name,source_path_binding_only=False,
                fresh_coupled_checkpoint_required=True,source_carlsim_seed=10,source_host_srand_seed=3,source_estdp_enabled=True))
        build['source_file_byte_checks'][0].update(source_path_binding_only=False,
                                                 coupled_extension_transformation_verified=True)
    write(root/'build_report.json', build)
    names = ('build_report.json', 'bin/ca3_retrieval', 'trainPat.csv', 'lib/libcarlsim.so')
    write(root/'manifest.json', {'files':[{'path':name, 'sha256':_sha(root/name)} for name in names]})
    native = {**build, 'status':family+'_RETRIEVAL_EXIT_ZERO', 'exit_code':0,
        'native_run_executed':True, 'retrieval_executed':True, 'expected_duration_ms':1001,
        'normalized_input_qualified':True, 'normalized_input_sha256':_sha(root/'normalized.dat'),
        'raw65p_sha256':'c'*64, 'formation_report_sha256':'d'*64, 'formation_replay_sha256':'e'*64,
        'normalization_report_sha256':'f'*64, 'normalization_replay_sha256':'1'*64,
        'runtime_manifest_sha256':_sha(root/'manifest.json'), 'slurm_job_id':'test-allocation',
        'native_pid':102 if repeat else 101,
        'started_utc':'2026-09-29T01:00:02+00:00' if repeat else '2026-09-29T01:00:00+00:00',
        'finished_utc':'2026-09-29T01:00:03+00:00' if repeat else '2026-09-29T01:00:01+00:00'}
    write(root/'native_report.json', native)
    validate_retrieval(root, root/'observations', coupled_extension=coupled_extension)


class RetrievalReplay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original, self.repeat = self.root/'original', self.root/'repeat'
        fixture(self.original)
        fixture(self.repeat, repeat=True)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, name='replay.json'):
        return replay.compare(self.original, self.repeat, self.root/name)

    def failure(self, match, name='failed.json', exception=ValueError):
        with self.assertRaisesRegex(exception, match):
            self.compare(name)
        row = json.loads((self.root/name).read_text())
        self.assertFalse(row['scientific_fidelity_qualified'])
        self.assertEqual(row['production_episodes'], 0)
        return row

    def replace_events(self, rows):
        events(self.repeat/'native_run/results/spk_E.dat', 3, rows)
        shutil.rmtree(self.repeat/'observations')
        validate_retrieval(self.repeat, self.repeat/'observations')

    def test_exact_events_and_independently_reconstructed_full_counts(self):
        row = self.compare()
        self.assertEqual(row['status'], 'PASS_NATIVE_RETRIEVAL_REPLAY_AND_COUNTS')
        self.assertEqual(row['strict_file_replay_status'], 'PASS_NATIVE_RETRIEVAL_FILE_BYTES')
        self.assertTrue(row['independent_fresh_process_replay'])
        self.assertTrue(row['actual_normalized_input_verified'])
        self.assertEqual(row['events_compared'], 7)
        self.assertEqual(row['independently_reconstructed_counts'], [dict(duration_ms=1001,
            events=7, unit_count_values=5, field_count_values=2002)]*2)
        self.assertFalse(row['native_internal_spike_counter_verified'])
        self.assertFalse(row['terminal_native_clock_independently_verified'])
        self.assertFalse(row['full_scale_qualified'])

    def test_same_time_order_and_duplicate_multiplicity_are_kept_separate(self):
        self.replace_events([(520,0),(520,2),(520,2),(1,0),(1000,1)])
        row = self.compare()
        self.assertTrue(row['event_multisets_equal'])
        self.assertTrue(row['same_time_file_order_differences'])
        self.assertFalse(row['event_file_bytes_equal'])
        self.assertEqual(row['strict_file_replay_status'], 'MISMATCH_NATIVE_RETRIEVAL_FILE_BYTES')
        self.assertEqual(row['status'], 'PASS_NATIVE_RETRIEVAL_REPLAY_AND_COUNTS')

    def test_timestamp_order_changes_are_retained(self):
        self.replace_events(list(reversed(E_ROWS)))
        row = self.compare()
        self.assertTrue(row['timestamp_order_differences'])
        self.assertFalse(row['timestamp_sequence_equal'])
        self.assertFalse(row['same_time_file_order_differences'])

    def test_matching_marginal_counts_cannot_hide_changed_time_unit_pairs(self):
        self.replace_events([(520,0),(520,0),(520,2),(1,2),(1000,1)])
        row = self.failure('exact event multiset')
        self.assertEqual(row['status'], 'MISMATCH_NATIVE_RETRIEVAL_REPLAY')
        self.assertFalse(row['event_multisets_equal'])

    def test_missing_duplicate_cannot_pass(self):
        self.replace_events(E_ROWS[1:])
        row = self.failure('exact event multiset')
        self.assertEqual(row['groups'][1]['original_rows'], 5)
        self.assertEqual(row['groups'][1]['repeated_rows'], 4)

    def test_rebound_wrong_counts_fail_independent_reconstruction(self):
        path = self.repeat/'observations/population_counts.npz'
        with np.load(path,allow_pickle=False) as data: values = {k:data[k] for k in data.files}
        values['population_counts_per_ms'][1000,1] += 1
        np.savez_compressed(path,**values)
        receipt = self.repeat/'observations/recording.json'
        data = json.loads(receipt.read_text()); data['observation_sha256'] = _sha(path); write(receipt,data)
        self.failure('independent event count projection differs')

    def test_noninteger_decoded_counts_rejected(self):
        path = self.repeat/'observations/population_counts.npz'
        with np.load(path,allow_pickle=False) as data: values = {k:data[k] for k in data.files}
        values['unit_counts'] = values['unit_counts'].astype(float)
        np.savez_compressed(path,**values)
        receipt = self.repeat/'observations/recording.json'
        data = json.loads(receipt.read_text()); data['observation_sha256'] = _sha(path); write(receipt,data)
        self.failure('counts must be integers')

    def test_malformed_array_archive_retains_failure_receipt(self):
        path = self.repeat/'observations/population_counts.npz'
        path.write_bytes(b'PK\x03\x04broken zip data')
        receipt = self.repeat/'observations/recording.json'
        data = json.loads(receipt.read_text()); data['observation_sha256'] = _sha(path); write(receipt, data)
        self.failure('File is not a zip file', exception=BadZipFile)

    def test_normalized_input_payload_and_missing_symlink_target_are_rejected(self):
        path = self.repeat/'normalized.dat'; before = path.read_bytes()
        path.write_bytes(before+b'changed')
        self.failure('normalized input differs', 'input_changed.json')
        path.unlink()
        self.failure('No such file', 'input_missing.json', FileNotFoundError)

    def test_changed_upstream_input_receipt_is_not_accepted(self):
        update_report(self.repeat, {'formation_replay_sha256':'2'*64})
        self.failure('qualified input changed: formation_replay_sha256')

    def test_malformed_upstream_receipt_hash_is_not_accepted(self):
        update_report(self.repeat, {'normalization_report_sha256':'not-a-sha'})
        self.failure('input receipt hash required')

    def test_incomplete_native_execution_and_stale_decoder_are_rejected(self):
        update_report(self.repeat, {'retrieval_executed':False})
        self.failure('completed pinned native retrieval', 'incomplete.json')
        update_report(self.repeat, {'retrieval_executed':True}, bind=False)
        self.failure('bound retrieval decoder', 'stale.json')

    def test_source_build_mutation_and_actual_binary_mutation_are_rejected(self):
        path = self.repeat/'build_report.json'
        data = json.loads(path.read_text()); data['integration']='changed'; write(path,data)
        refresh_manifest(self.repeat)
        self.failure('differs from source build', 'build.json')
        data['integration']='RUNGE_KUTTA4_5'; write(path,data); refresh_manifest(self.repeat)
        (self.repeat/'bin/ca3_retrieval').write_bytes(b'changed')
        self.failure('runtime input differs from manifest', 'binary.json')

    def test_rebound_changed_source_revisions_are_rejected(self):
        update_report(self.repeat, {'author_revision':'changed'})
        self.failure('completed pinned native retrieval')

    def test_runtime_manifest_duplicate_path_and_coverage_are_rejected(self):
        path = self.repeat/'manifest.json'; data = json.loads(path.read_text()); original=list(data['files'])
        data['files'].append(data['files'][0]); write(path,data)
        update_report(self.repeat, {'runtime_manifest_sha256':_sha(path)})
        self.failure('manifest path or digest invalid', 'duplicate.json')
        data['files'] = [row for row in original if row['path'] != 'bin/ca3_retrieval']; write(path,data)
        update_report(self.repeat, {'runtime_manifest_sha256':_sha(path)})
        self.failure('manifest coverage required', 'coverage.json')

    def test_decoder_group_file_coverage_and_denominators_are_required(self):
        path = self.repeat/'observations/recording.json'; before=path.read_text()
        data=json.loads(before);data['groups'][0]['start']=1;write(path,data)
        self.failure('decoder group identity', 'groups.json')
        data=json.loads(before);data['files']=[data['files'][0]]*2;write(path,data)
        self.failure('event file coverage', 'files.json')
        data=json.loads(before);data['files'][0]['records']=99;write(path,data)
        self.failure('event file denominator differs', 'denominator.json')

    def test_independent_process_and_report_binding_are_required(self):
        old=json.loads((self.repeat/'native_report.json').read_text())
        left=json.loads((self.original/'native_report.json').read_text())
        update_report(self.repeat, {k:left[k] for k in ('native_pid','started_utc','finished_utc')})
        self.failure('independent subsequent retrieval process', 'copied.json')
        update_report(self.repeat, {k:old[k] for k in ('native_pid','started_utc','finished_utc')})
        update_report(self.repeat, {'independent_repeat_of_report_sha256':'0'*64})
        self.failure('different original retrieval', 'wrong_original.json')

    def test_invalid_process_metadata_and_timezone_are_rejected(self):
        update_report(self.repeat, {'native_pid':0})
        self.failure('process identity required', 'pid.json')
        update_report(self.repeat, {'native_pid':102,'started_utc':'2026-09-29T01:00:02'})
        self.failure('timestamp timezone required', 'timezone.json')

    def test_same_runtime_is_rejected_and_existing_evidence_is_preserved(self):
        with self.assertRaisesRegex(ValueError,'distinct fresh retrieval runtime'):
            replay.compare(self.original,self.original,self.root/'same.json')
        self.compare(); before=(self.root/'replay.json').read_bytes()
        with self.assertRaises(FileExistsError): self.compare()
        self.assertEqual((self.root/'replay.json').read_bytes(), before)

    def test_event_header_change_is_not_hidden_by_equal_pairs(self):
        path=self.repeat/'native_run/results/spk_E.dat'
        path.write_bytes(SPIKE_HEADER.pack(SPIKE_SIGNATURE,.2,1,3,1)+path.read_bytes()[20:])
        receipt=self.repeat/'observations/recording.json'
        data=json.loads(receipt.read_text());data['files'][1]['sha256']=_sha(path);write(receipt,data)
        row=self.failure('exact event multiset or event header')
        self.assertFalse(row['groups'][1]['header_bytes_equal'])
        self.assertTrue(row['groups'][1]['event_multiset_equal'])

    def test_receipt_changes_during_replay_are_rejected(self):
        original_counts=replay._independent_counts
        def changed(root,header,recording):
            result=original_counts(root,header,recording)
            if root==self.repeat:
                path=root/'observations/recording.json'
                data=json.loads(path.read_text());data['changed']=True;write(path,data)
            return result
        with patch.object(replay,'_independent_counts',side_effect=changed):
            self.failure('changed during replay')

    def test_cli_success_and_failure_exit_codes_and_artifacts(self):
        base=[sys.executable,'-m','population_dynamics.carlsim_retrieval_replay',
              '--original',str(self.original),'--repeat',str(self.repeat)]
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        good=subprocess.run([*base,'--output',str(self.root/'cli_pass.json')],capture_output=True,text=True,env=env)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(json.loads(good.stdout)['status'],'PASS_NATIVE_RETRIEVAL_REPLAY_AND_COUNTS')
        update_report(self.repeat,{'formation_report_sha256':'0'*64})
        bad=subprocess.run([*base,'--output',str(self.root/'cli_fail.json')],capture_output=True,text=True,env=env)
        self.assertEqual(bad.returncode,2)
        self.assertIn('formation_report_sha256',bad.stderr)
        self.assertEqual(json.loads((self.root/'cli_fail.json').read_text())['status'],'FAILED_NATIVE_RETRIEVAL_REPLAY')


class CoupledRetrievalReplay(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.original,self.repeat=self.root/'coupled-original',self.root/'coupled-repeat'
        fixture(self.original,coupled_extension=True)
        fixture(self.repeat,repeat=True,coupled_extension=True)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self,name='coupled.json',*,coupled_extension=True,original=None,repeat=None):
        return replay.compare(original or self.original,repeat or self.repeat,self.root/name,
                              coupled_extension=coupled_extension)

    def failure(self,match,name='failed.json',**kwargs):
        with self.assertRaisesRegex(ValueError,match):
            self.compare(name,**kwargs)
        report=json.loads((self.root/name).read_text())
        self.assertFalse(report['scientific_fidelity_qualified'])
        self.assertEqual(report['production_episodes'],0)
        return report

    def rebind_identity(self,key,value):
        build=json.loads((self.repeat/'build_report.json').read_text())
        build[key]=value;write(self.repeat/'build_report.json',build)
        update_report(self.repeat,{key:value})
        refresh_manifest(self.repeat)
        if key=='coupled_extension':
            path=self.repeat/'observations/recording.json'
            recording=json.loads(path.read_text());recording[key]=value;write(path,recording)

    def test_sparse_full_population_replays_events_and_integer_counts(self):
        report=self.compare()
        self.assertEqual(report['status'],'PASS_COUPLED_EXTENSION_RETRIEVAL_REPLAY_AND_COUNTS')
        self.assertEqual(report['strict_file_replay_status'],'PASS_COUPLED_EXTENSION_RETRIEVAL_FILE_BYTES')
        self.assertEqual(report['units'],168106)
        self.assertEqual(len(report['groups']),8)
        self.assertEqual(report['events_compared'],19)
        self.assertEqual(report['independently_reconstructed_counts'],[dict(
            duration_ms=1001,events=19,unit_count_values=168106,field_count_values=8008)]*2)
        self.assertTrue(report['independent_fresh_process_replay'])
        self.assertTrue(report['actual_normalized_input_verified'])
        self.assertEqual(report['coupled_extension']['source_role'],'retrieval')
        self.assertEqual(report['coupled_extension']['derived_cue_size'],276)
        self.assertFalse(report['full_scale_qualified'])
        self.assertFalse(report['pattern_completion_qualified'])
        recording=json.loads((self.repeat/'observations/recording.json').read_text())
        self.assertEqual(recording['status'],'DECODED_COUPLED_EXTENSION_RETRIEVAL_FILES')

    def test_explicit_selection_and_mixed_families_are_enforced(self):
        result=self.failure('completed pinned native retrieval','default.json',coupled_extension=False)
        self.assertEqual(result['status'],'FAILED_NATIVE_RETRIEVAL_REPLAY')
        native=self.root/'native';fixture(native)
        for name,left,right in [('native-left.json',native,self.repeat),('native-right.json',self.original,native)]:
            result=self.failure('CA3_COUPLED_SCOPE_REQUIRED',name,original=left,repeat=right)
            self.assertEqual(result['status'],'FAILED_COUPLED_EXTENSION_RETRIEVAL_REPLAY')

    def test_complete_extension_and_config_identity_must_match_between_runs(self):
        extension=json.loads((self.repeat/'native_report.json').read_text())['coupled_extension']
        extension['source_pattern_sha256']='7'*64
        self.rebind_identity('coupled_extension',extension)
        self.failure('coupled retrieval source changed: coupled_extension','extension.json')
        extension['source_pattern_sha256']='9'*64
        self.rebind_identity('coupled_extension',extension)
        config=dict(resolved={'population_multiplier':2},sha256='7'*64)
        self.rebind_identity('coupled_extension_config',config)
        self.failure('coupled retrieval source changed: coupled_extension_config','config.json')

    def test_missing_fresh_checkpoint_declaration_is_rejected(self):
        original=json.loads((self.repeat/'native_report.json').read_text())['retrieval_input_contract']
        for number,value in enumerate((None,False)):
            contract=dict(original)
            if value is None:
                del contract['fresh_coupled_checkpoint_required']
            else:
                contract['fresh_coupled_checkpoint_required']=value
            self.rebind_identity('retrieval_input_contract',contract)
            self.failure('retrieval input binding required',f'fresh-{number}.json')

    def test_equal_marginals_do_not_hide_changed_time_unit_pairs(self):
        units=coupled_declaration()['derived_populations']['CA3_Pyramidal']
        rows=[(520,0),(520,0),(520,2),(1,2),(1000,units-1)]
        events(self.repeat/'native_run/results/spk_CA3_Pyramidal.dat',units,rows)
        shutil.rmtree(self.repeat/'observations')
        validate_retrieval(self.repeat,self.repeat/'observations',coupled_extension=True)
        report=self.failure('exact event multiset')
        self.assertEqual(report['status'],'MISMATCH_COUPLED_EXTENSION_RETRIEVAL_REPLAY')
        self.assertFalse(report['event_multisets_equal'])

    def test_rebound_counts_and_decoder_extension_cannot_hide_inconsistency(self):
        receipt=self.repeat/'observations/recording.json'
        original=json.loads(receipt.read_text())
        changed=json.loads(json.dumps(original))
        changed['coupled_extension']['derived_cue_size']=550
        write(receipt,changed)
        self.failure('identity differs across build/execution/decoder','decoder-identity.json')
        write(receipt,original)
        path=self.repeat/'observations/population_counts.npz'
        with np.load(path,allow_pickle=False) as saved:
            arrays={key:saved[key] for key in saved.files}
        arrays['population_counts_per_ms'][1000,-1]+=1
        np.savez_compressed(path,**arrays)
        original['observation_sha256']=_sha(path);write(receipt,original)
        self.failure('independent event count projection differs','counts.json')

    def test_duplicate_process_metadata_is_not_independent_retrieval(self):
        left=json.loads((self.original/'native_report.json').read_text())
        update_report(self.repeat,{key:left[key] for key in ('native_pid','started_utc','finished_utc')})
        self.failure('independent subsequent retrieval process')

    def test_cli_coupled_flag_preserves_family_and_default_rejection(self):
        command=[sys.executable,'-m','population_dynamics.carlsim_retrieval_replay',
                 '--original',str(self.original),'--repeat',str(self.repeat)]
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        good=subprocess.run([*command,'--coupled-extension','--output',str(self.root/'cli-coupled.json')],
                            capture_output=True,text=True,env=env)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(json.loads(good.stdout)['status'],'PASS_COUPLED_EXTENSION_RETRIEVAL_REPLAY_AND_COUNTS')
        bad=subprocess.run([*command,'--output',str(self.root/'cli-default.json')],capture_output=True,text=True,env=env)
        self.assertEqual(bad.returncode,2)
        self.assertEqual(json.loads((self.root/'cli-default.json').read_text())['status'],'FAILED_NATIVE_RETRIEVAL_REPLAY')



def observation_fixture(root, header, *, change=None, header_only=False):
    """Real observation binary; tiny streams exercise all1001ms without GB fixtures."""
    path = root/'native_observation.bin'
    groups = header['groups']
    with path.open('wb') as stream:
        stream.write(observation.HEADER.pack(observation.MAGIC, len(groups), header['units']))
        for group in groups:
            name = group.name.encode()
            local_group = group.group_id+(10 if change == 'mapping' else 0)
            stream.write(observation.GROUP.pack(
                group.group_id, group.start, group.stop, 0, local_group,
                group.start, group.stop, 1, len(name))+name)
        if header_only:
            return path
        for ms in range(replay.DURATION_MS):
            for kind in (b'R', b'A', b'C'):
                for group in groups:
                    values = np.zeros(group.units, dtype='<i4' if kind == b'C' else '<f4')
                    if ms == 1000 and group.group_id == 0:
                        if ((change == 'current_bits' and kind in (b'R', b'A'))
                                or (change == 'requested_bits' and kind == b'R')
                                or (change == 'effective_bits' and kind == b'A')):
                            values.view('<u4')[0] = 0x80000000
                        if change == 'counter' and kind == b'C':
                            values[0] = 1
                    start = ms-1 if change == 'time' and ms == 1000 and kind == b'A' else ms
                    stop = ms if kind == b'R' else ms+1
                    stream.write(observation.RECORD.pack(kind, start, stop, group.group_id, group.units))
                    stream.write(values.tobytes())
        if change != 'missing_footer':
            stop = 1000 if change == 'footer' else replay.DURATION_MS
            stream.write(observation.RECORD.pack(b'E', stop, stop, -1, 0))
    return path


def refinement_fixture(root, count):
    """Real full-size identity metadata with a header-only observation, never executed."""
    build = json.loads((root/'build_report.json').read_text())
    values = dict(integration='RUNGE_KUTTA4_'+str(count),
                  numerical_refinement=substeps.contract(count),
                  numerical_refinement_implementation_sha256=_sha(Path(substeps.__file__)),
                  overlay_source_sha256=substeps.MAIN_SHA256[count],
                  native_observation_overlay=dict(native_execution_verified=False, fixture=True),
                  conductance_memory_overlay=dict(fixture=True), retrieval_condition='source')
    build.update(values); write(root/'build_report.json', build)
    header = replay.network_header(root/'normalized.dat')
    trace = observation_fixture(root, header, header_only=True)
    update_report(root, {**values,
        'native_observation_overlay':dict(native_execution_verified=True, fixture=True),
        'native_observation_sha256':_sha(trace), 'native_observation_bytes':trace.stat().st_size,
        'native_observation_stream_validated':True,
        'observed_native_duration_ms':1001, 'host_request_runtime_current_equal':True})
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['native_observation_enabled'] = True; write(root/'manifest.json',manifest)
    refresh_manifest(root)
    shutil.rmtree(root/'observations')
    validate_retrieval(root, root/'observations', coupled_extension=True, neuron_substeps=count)


class RefinedRetrievalBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root/'coupled', coupled_extension=True)
        self.runtime = self.root/'coupled'

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_each_supported_count_binds_actual_inputs_and_observation_header(self):
        for count in substeps.COUNTS:
            with self.subTest(count=count):
                refinement_fixture(self.runtime, count)
                bound = replay._bind(self.runtime, coupled_extension=True, neuron_substeps=count)
                self.assertEqual(bound.report['numerical_refinement'], substeps.contract(count))
                self.assertEqual(bound.hashes['native_observation.bin'],
                                 _sha(self.runtime/'native_observation.bin'))
                self.assertEqual(bound.header['units'], 168106)

    def test_missing_or_wrong_selection_rejects_refinements_and_non_coupled_call(self):
        refinement_fixture(self.runtime, 20)
        for count,error in ((None,'EXPLICIT_MATCHING_SUBSTEPS'),
                            (10,'BUILD_OR_REQUEST_DIFFERS'), (7,'MUST_BE_5_10_20_OR_40')):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError,error):
                replay._bind(self.runtime,coupled_extension=True,neuron_substeps=count)
        with self.assertRaisesRegex(ValueError,'REQUIRES_COUPLED_SOURCE_RETRIEVAL'):
            replay._bind(self.runtime,neuron_substeps=20)

    def test_missing_or_mismatching_metadata_is_rejected(self):
        refinement_fixture(self.runtime,20)
        for values in ({'numerical_refinement':None},
                       {'numerical_refinement':substeps.contract(10)},
                       {'retrieval_condition':'cue_omission'},
                       {'numerical_refinement_implementation_sha256':'0'*64}):
            with self.subTest(values=values):
                before=(self.runtime/'native_report.json').read_text()
                update_report(self.runtime,values)
                with self.assertRaisesRegex(ValueError,'BUILD_OR_REQUEST_DIFFERS'):
                    replay._bind(self.runtime,coupled_extension=True,neuron_substeps=20)
                update_report(self.runtime,json.loads(before))
        receipt=self.runtime/'observations/recording.json'
        row=json.loads(receipt.read_text());row['integration']='RUNGE_KUTTA4_10';write(receipt,row)
        with self.assertRaisesRegex(ValueError,'decoder numerical declaration differs'):
            replay._bind(self.runtime,coupled_extension=True,neuron_substeps=20)

    def test_actual_observation_hash_bytes_and_execution_flag_are_bound(self):
        refinement_fixture(self.runtime,20)
        trace=self.runtime/'native_observation.bin'
        trace.write_bytes(trace.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'observation differs from native receipt'):
            replay._bind(self.runtime,coupled_extension=True,neuron_substeps=20)
        update_report(self.runtime,{'native_observation_sha256':_sha(trace)})
        with self.assertRaisesRegex(ValueError,'observation differs from native receipt'):
            replay._bind(self.runtime,coupled_extension=True,neuron_substeps=20)
        update_report(self.runtime,{'native_observation_bytes':trace.stat().st_size,
                                   'native_observation_stream_validated':False})
        with self.assertRaisesRegex(ValueError,'completed bound native observation'):
            replay._bind(self.runtime,coupled_extension=True,neuron_substeps=20)

    def test_legacy_helper_binds_existing_counts_but_cannot_authorize40(self):
        for count in substeps.COUNTS:
            with self.subTest(count=count):
                refinement_fixture(self.runtime,count)
                build=json.loads((self.runtime/'build_report.json').read_text())
                build['numerical_refinement_implementation_sha256']=substeps.LEGACY_IMPLEMENTATION_SHA256
                write(self.runtime/'build_report.json',build)
                update_report(self.runtime,{'numerical_refinement_implementation_sha256':
                                            substeps.LEGACY_IMPLEMENTATION_SHA256})
                refresh_manifest(self.runtime)
                if count == 40:
                    with self.assertRaisesRegex(ValueError,'BUILD_OR_REQUEST_DIFFERS'):
                        replay._bind(self.runtime,coupled_extension=True,neuron_substeps=count)
                else:
                    bound=replay._bind(self.runtime,coupled_extension=True,neuron_substeps=count)
                    self.assertEqual(bound.report['integration'],'RUNGE_KUTTA4_'+str(count))

    def test_cli_explicit_option_reaches_stream_validation_and_default_rejects(self):
        for count in (20,40):
            with self.subTest(count=count):
                refinement_fixture(self.runtime,count)
                other=self.root/('repeat-'+str(count))
                fixture(other,repeat=True,coupled_extension=True);refinement_fixture(other,count)
                command=[sys.executable,'-m','population_dynamics.carlsim_retrieval_replay',
                         '--original',str(self.runtime),'--repeat',str(other),'--coupled-extension']
                env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
                explicit=subprocess.run([*command,'--neuron-substeps',str(count),
                                         '--output',str(self.root/('explicit-'+str(count)+'.json'))],
                                        capture_output=True,text=True,env=env)
                self.assertEqual(explicit.returncode,2)
                # Header-only full-size fixture binds, then fails actual stream parsing.
                self.assertIn('CA3_OBSERVATION_TRUNCATED',explicit.stderr)
                default=subprocess.run([*command,'--output',str(self.root/('default-'+str(count)+'.json'))],
                                       capture_output=True,text=True,env=env)
                self.assertEqual(default.returncode,2)
                self.assertIn('EXPLICIT_MATCHING_SUBSTEPS',default.stderr)


class RefinedObservationReplay(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.original,self.repeat=self.root/'original',self.root/'repeat'
        self.bound=[]
        for root,repeated in ((self.original,False),(self.repeat,True)):
            fixture(root,repeat=repeated)
            bound=replay._bind(root)
            trace=observation_fixture(root,bound.header)
            bound.hashes['native_observation.bin']=_sha(trace)
            self.bound.append(bound)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_intervals_and_terminal1001ms_are_compared(self):
        result=replay._compare_native_observations(*self.bound)
        self.assertEqual(result['intervals_compared'],1001)
        self.assertEqual(result['values_compared_per_array'],5005)
        self.assertEqual(result['terminal_time_ms'],1001)
        self.assertTrue(result['requested_current_bits_equal'])
        self.assertTrue(result['effective_current_bits_equal'])
        self.assertTrue(result['native_counter_arrays_equal'])
        self.assertFalse(result['complete_microstate_equality_qualified'])
        self.assertFalse(result['whole_simulator_convergence_qualified'])

    def test_changed_float_bits_counters_clock_and_footer_are_rejected(self):
        for change,error in (('current_bits','requested_current bits changed at ms 1000'),
                             ('requested_bits','REQUEST_RUNTIME_CURRENT_MISMATCH'),
                             ('effective_bits','REQUEST_RUNTIME_CURRENT_MISMATCH'),
                             ('counter','counter array changed at ms 1000'),
                             ('time','INTERVAL_ORDER_OR_COVERAGE_INVALID'),
                             ('footer','DURATION_OR_FOOTER_INVALID'),
                             ('missing_footer','TRUNCATED'),
                             ('mapping','runtime group mapping changed')):
            with self.subTest(change=change):
                observation_fixture(self.repeat,self.bound[1].header,change=change)
                with self.assertRaisesRegex(ValueError,error):
                    replay._compare_native_observations(*self.bound)

    def test_explicit_comparison_combines_real_events_and_real_observation_stream(self):
        # Only _bind is mocked to substitute five-unit runs for168106-unit coupled
        # streams. RefinedRetrievalBinding separately exercises real full-size binding.
        # No event/count projection, observation parser, hash or clock check is mocked.
        for bound in self.bound:
            bound.report.update(numerical_refinement=substeps.contract(20),
                                integration='RUNGE_KUTTA4_20',
                                coupled_extension=coupled_declaration(),
                                coupled_extension_config=dict(resolved={'population_multiplier':2},sha256='8'*64))
        by_root={bound.root:bound for bound in self.bound}
        with patch.object(replay,'_bind',side_effect=lambda root,**kwargs:by_root[root]) as binding:
            result=replay.compare(self.original,self.repeat,self.root/'refined.json',
                                  coupled_extension=True,neuron_substeps=20)
        self.assertEqual(binding.call_count,2)
        self.assertEqual(binding.call_args.kwargs,dict(coupled_extension=True,neuron_substeps=20))
        self.assertEqual(result['status'],'PASS_COUPLED_EXTENSION_RETRIEVAL_REPLAY_AND_COUNTS')
        self.assertTrue(result['native_counter_replay_equal'])
        self.assertTrue(result['actual_current_trace_verified'])
        self.assertTrue(result['terminal_native_clock_independently_verified'])
        self.assertFalse(result['native_internal_spike_counter_verified'])
        self.assertTrue(result['event_multisets_equal'])
        self.assertFalse(result['scientific_fidelity_qualified'])


if __name__ == '__main__':
    unittest.main()
