import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import carlsim_formation_replay as replay
from .carlsim_records import (CA3_REFERENCE_POPULATIONS, GROUP_HEADER, NETWORK_HEADER,
                              NETWORK_SIGNATURE, SOURCE_REVISION, _sha,
                              validate_formation, validate_prefix)
from .test_carlsim_prefix_replay import fixture
from .test_carlsim_records import events, network

SOURCE = dict(integration='RUNGE_KUTTA4_5', delay_ms=1, source_rng='source_seed10',
              pattern_sha256='c'*64, pattern_integer_count=1100)
PREFIX_E = [(20,2),(20,2),(2,0),(4449,1)]


def update_report(root, values, *, bind=True):
    path = root/'native_report.json'
    report = json.loads(path.read_text())
    report.update(values)
    path.write_text(json.dumps(report))
    if bind:
        receipt = root/'observations/recording.json'
        data = json.loads(receipt.read_text())
        data['native_report_sha256'] = _sha(path)
        receipt.write_text(json.dumps(data))


def original_fixture(root, *, e_rows=None):
    fixture(root)
    update_report(root, SOURCE)
    if e_rows is not None:
        events(root/'native_run/results/spk_E.dat',3,e_rows)
        shutil.rmtree(root/'observations')
        validate_prefix(root,root/'observations')


def formation_fixture(root, original, *, e_rows=None, wall_time=9., corrupt_prefix=False):
    run = root/'native_run'
    (run/'results').mkdir(parents=True)
    payload = bytearray((original/'native_run/network5p.dat').read_bytes())
    payload[12:16] = struct.pack('<f',wall_time)
    if corrupt_prefix:
        payload[-1] ^= 1
    (run/'network5p.dat').write_bytes(payload)
    network(run/'network100p.dat', seconds=70.)
    with (run/'network100p.dat').open('ab') as stream:
        stream.write(b'terminal-synaptic-payload')
    # Future records precede valid earlier timestamps; filtering cannot stop early.
    events(run/'results/spk_I.dat',2,[(4450,1),(0,0),(69999,0),(12,1)])
    e_rows = PREFIX_E if e_rows is None else e_rows
    events(run/'results/spk_E.dat',3,[(69999,2),*e_rows,(4450,1)])
    report = json.loads((original/'native_report.json').read_text())
    report.update(status='NATIVE_FORMATION_EXIT_ZERO', scope='native_source_formation',
        expected_duration_ms=70000, complete_formation_executed=True,
        complete_author_formation_program_unchanged=True,
        original_author_main_sha256='f'*64, overlay_source_sha256='f'*64,
        binary_sha256='b'*64, checkpoint_sha256=_sha(run/'network100p.dat'))
    (root/'native_report.json').write_text(json.dumps(report))
    validate_formation(root,root/'observations')


class FormationReplay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original,self.formation = self.root/'original',self.root/'formation'
        original_fixture(self.original)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, name='replay.json'):
        return replay.compare(self.original,self.formation,self.root/name,workdir=self.root)

    def failure(self, match, name='failed.json'):
        with self.assertRaisesRegex(ValueError,match):
            self.compare(name)
        result = json.loads((self.root/name).read_text())
        self.assertFalse(result['scientific_fidelity_qualified'])
        return result

    def test_full_counts_and_half_open_prefix_with_out_of_order_future_events(self):
        formation_fixture(self.formation,self.original)
        r = self.compare()
        self.assertEqual(r['status'],'PASS_NATIVE_FORMATION_PREFIX_REPLAY_AND_COUNTS')
        self.assertEqual(r['prefix_events_compared'],6)
        self.assertEqual([x['events'] for x in r['independently_reconstructed_counts']],[6,10])
        self.assertEqual(r['independently_reconstructed_counts'][1]['field_count_values'],140000)
        self.assertEqual(sum(g['future_records'] for g in r['groups']),4)
        self.assertEqual(r['checks'][0]['ignored_byte_range'],[12,16])
        self.assertNotEqual(r['executable_hashes']['prefix'],r['executable_hashes']['formation'])
        self.assertFalse(r['independent_full_formation_execution_replay'])
        self.assertFalse(r['complete_formation_retrieval'])
        self.assertFalse(r['full_scale_qualified'])
        self.assertFalse(list(self.root.glob('carlsim-formation-prefix-*')))

    def test_same_time_order_changes_remain_descriptive_and_keep_multiplicity(self):
        shutil.rmtree(self.original)
        original_fixture(self.original,e_rows=[(20,0),(20,2),(20,0),(4449,1)])
        formation_fixture(self.formation,self.original,e_rows=[(20,2),(20,0),(20,0),(4449,1)])
        r = self.compare()
        self.assertTrue(r['prefix_event_multisets_equal'])
        self.assertTrue(r['same_time_file_order_differences'])
        self.assertFalse(r['prefix_event_order_equal'])

    def test_timestamp_order_changes_do_not_change_exact_pairs(self):
        formation_fixture(self.formation,self.original,e_rows=[(20,2),(2,0),(20,2),(4449,1)])
        r = self.compare()
        self.assertTrue(r['timestamp_order_differences'])
        self.assertFalse(r['same_time_file_order_differences'])

    def test_equal_marginals_cannot_hide_changed_prefix_time_id_pairs(self):
        formation_fixture(self.formation,self.original,e_rows=[(20,2),(20,0),(2,2),(4449,1)])
        r = self.failure('exact event multiset differs')
        self.assertEqual(r['status'],'MISMATCH_NATIVE_FORMATION_PREFIX_REPLAY')
        self.assertFalse(r['prefix_event_multisets_equal'])

    def test_duplicate_multiplicity_and_missing_prefix_records_fail(self):
        for i,rows in enumerate(([(20,2),(2,0),(2,0),(4449,1)],[(20,2),(2,0),(4449,1)])):
            with self.subTest(i=i):
                if self.formation.exists():shutil.rmtree(self.formation)
                formation_fixture(self.formation,self.original,e_rows=rows)
                self.failure('exact event multiset differs',f'multiplicity_{i}.json')

    def test_checkpoint_payload_change_is_not_hidden_by_ignored_wall_time(self):
        formation_fixture(self.formation,self.original,corrupt_prefix=True)
        r = self.failure('checkpoint or exact event multiset differs')
        self.assertFalse(r['checks'][0]['equal'])
        self.assertTrue(r['prefix_event_multisets_equal'])

    def test_changed_source_with_rebound_receipt_still_fails(self):
        formation_fixture(self.formation,self.original)
        for key,value in [('author_revision','changed'),('integration','changed'),('pattern_sha256','d'*64)]:
            with self.subTest(key=key):
                old = json.loads((self.formation/'native_report.json').read_text())[key]
                update_report(self.formation,{key:value})
                self.failure('native source changed: '+key,key+'.json')
                update_report(self.formation,{key:old})

    def test_memory_overlay_identity_must_match_both_executions(self):
        formation_fixture(self.formation,self.original)
        declaration = {"overlay_id":"ca3_conductance_memory", "initialization_correction":"full rows"}
        update_report(self.formation, {"conductance_memory_overlay":declaration})
        self.failure('conductance memory policies differ','mixed-memory.json')
        update_report(self.original, {"conductance_memory_overlay":declaration})
        result = self.compare('same-memory.json')
        self.assertEqual(result['status'],'PASS_NATIVE_FORMATION_PREFIX_REPLAY_AND_COUNTS')
        self.assertFalse(result['scientific_fidelity_qualified'])

    def test_stale_native_receipt_and_changed_event_source_fail(self):
        formation_fixture(self.formation,self.original)
        update_report(self.formation,{'extra':'changed'},bind=False)
        self.failure('bound decoder receipt required','stale.json')
        update_report(self.formation,{},bind=True)
        events(self.formation/'native_run/results/spk_E.dat',3,[(0,0)])
        self.failure('native spike file differs','changed_events.json')

    def test_rebound_wrong_full_counts_fail_independent_reconstruction(self):
        formation_fixture(self.formation,self.original)
        path = self.formation/'observations/population_counts.npz'
        with np.load(path,allow_pickle=False) as saved:
            data = {key:saved[key] for key in saved.files}
        data['population_counts_per_ms'][69999,0] += 1
        np.savez_compressed(path,**data)
        receipt = self.formation/'observations/recording.json'
        row = json.loads(receipt.read_text());row['observation_sha256']=_sha(path)
        receipt.write_text(json.dumps(row))
        self.failure('independent event count projection differs')

    def test_decoder_file_coverage_and_group_identity_are_required(self):
        formation_fixture(self.formation,self.original)
        receipt = self.formation/'observations/recording.json'
        row = json.loads(receipt.read_text());before=dict(row)
        row['files']=[row['files'][0],row['files'][0]]
        receipt.write_text(json.dumps(row))
        self.failure('decoder event file coverage differs','coverage.json')
        before['groups'][0]['name']='changed'
        receipt.write_text(json.dumps(before))
        self.failure('decoder group identity differs','groups.json')

    def test_incomplete_formation_and_changed_terminal_checkpoint_fail(self):
        formation_fixture(self.formation,self.original)
        update_report(self.formation,{'complete_formation_executed':False})
        self.failure('complete unchanged author formation required','incomplete.json')
        update_report(self.formation,{'complete_formation_executed':True})
        path = self.formation/'native_run/network100p.dat'
        with path.open('ab') as stream:stream.write(b'changed')
        self.failure('terminal checkpoint differs','terminal.json')

    def test_receipt_changes_during_replay_are_detected(self):
        formation_fixture(self.formation,self.original)
        original_counts = replay._independent_counts
        def counts(root,header,recording):
            result = original_counts(root,header,recording)
            if root == self.formation:
                path = root/'observations/recording.json'
                value = json.loads(path.read_text());value['changed_during_replay']=True
                path.write_text(json.dumps(value))
            return result
        with patch.object(replay,'_independent_counts',side_effect=counts):
            self.failure('receipt changed during replay')

    def test_cli_success_and_failure_have_correct_exit_codes_and_artifacts(self):
        formation_fixture(self.formation,self.original)
        base = [sys.executable,'-m','population_dynamics.carlsim_formation_replay',
                '--original',str(self.original),'--formation',str(self.formation),'--workdir',str(self.root)]
        env = {**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        good = subprocess.run([*base,'--output',str(self.root/'cli_pass.json')],capture_output=True,text=True,env=env)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(json.loads(good.stdout)['status'],'PASS_NATIVE_FORMATION_PREFIX_REPLAY_AND_COUNTS')
        update_report(self.formation,{'author_revision':'changed'})
        bad = subprocess.run([*base,'--output',str(self.root/'cli_fail.json')],capture_output=True,text=True,env=env)
        self.assertEqual(bad.returncode,2)
        self.assertIn('native source changed',bad.stderr)
        self.assertEqual(json.loads((self.root/'cli_fail.json').read_text())['status'],'FAILED_NATIVE_FORMATION_REPLAY')

    def test_existing_evidence_is_not_overwritten(self):
        formation_fixture(self.formation,self.original)
        self.compare()
        before = (self.root/'replay.json').read_bytes()
        with self.assertRaises(FileExistsError):self.compare()
        self.assertEqual((self.root/'replay.json').read_bytes(),before)


# Sparse native-format fixtures retain the real coupled unit count. They are
# software tests, not outputs from a mechanistic simulation or learned network.
def coupled_declaration():
    source = dict(CA3_REFERENCE_POPULATIONS)
    derived = {name: 2 * count for name, count in source.items()}
    return dict(schema='ca3_coupled_extension_v1',
        status='PROSPECTIVE_UNQUALIFIED_COUPLED_EXTENSION', multiplier=2,
        source_role='formation', source_core_neurons=84053, derived_core_neurons=168106,
        source_populations=source, derived_populations=derived,
        source_assembly_size=275, derived_assembly_size=550,
        source_cue_size=275, derived_cue_size=550,
        fresh_native_graph_required=True, fresh_formation_and_normalized_checkpoint_required=True,
        graph_or_output_cloning=False, source_pattern_sha256='9'*64,
        derived_pattern_sha256=SOURCE['pattern_sha256'], scientific_fidelity_qualified=False)


def coupled_network(path, *, seconds):
    populations = coupled_declaration()['derived_populations']
    payload = NETWORK_HEADER.pack(NETWORK_SIGNATURE, .3, seconds, 1., 168106, len(populations))
    start = 0
    for name, count in populations.items():
        payload += GROUP_HEADER.pack(start, start + count - 1, count, 1, 1, name.encode())
        start += count
    if start != 168106:
        raise ValueError('coupled fixture must contain the complete declared unit population')
    path.write_bytes(payload + b'sparse-test-checkpoint-payload')


def coupled_fixture(root, original=None, *, corrupt_prefix=False, changed_pairs=False):
    formation = original is not None
    run = root/'native_run'
    (run/'results').mkdir(parents=True)
    if formation:
        prefix = bytearray((original/'native_run/network5p.dat').read_bytes())
        prefix[12:16] = struct.pack('<f', 9.)
        if corrupt_prefix:
            prefix[-1] ^= 1
        (run/'network5p.dat').write_bytes(prefix)
        coupled_network(run/'network100p.dat', seconds=70.)
    else:
        coupled_network(run/'network5p.dat', seconds=4.45)
    extension = coupled_declaration()
    for name, count in extension['derived_populations'].items():
        rows = [(0,0), (4449,count-1)]
        if name == 'CA3_Pyramidal':
            rows = [(20,2), (20,2), (2,0), (4449,count-1)]
            if changed_pairs:
                rows = [(20,2), (20,0), (2,2), (4449,count-1)]
        if formation:
            rows = [(69999,0), *rows, (4450,count-1)]
        events(run/'results'/('spk_'+name+'.dat'), count, rows)
    checkpoint = run/('network100p.dat' if formation else 'network5p.dat')
    report = dict(SOURCE, status='COUPLED_EXTENSION_FORMATION_EXIT_ZERO' if formation else 'COUPLED_EXTENSION_PREFIX_EXIT_ZERO',
        scope='coupled_extension_formation' if formation else 'coupled_extension_prefix',
        framework_revision=SOURCE_REVISION, author_revision='sparse-fixture-author',
        exit_code=0, native_run_executed=True, core_neurons=168106,
        populations=extension['derived_populations'], expected_prefix_ms=4450,
        expected_duration_ms=70000 if formation else 4450,
        complete_formation_executed=formation, complete_author_formation_program_unchanged=False,
        source_code_semantics_preserved_before_stop=False,
        original_author_main_sha256='f'*64, overlay_source_sha256='e'*64 if formation else 'd'*64,
        binary_sha256='b'*64 if formation else 'a'*64,
        checkpoint_sha256=_sha(checkpoint), coupled_extension=extension,
        coupled_extension_config=dict(resolved={'population_multiplier':2}, sha256='8'*64))
    report['pattern_integer_count'] = 2200
    (root/'native_report.json').write_text(json.dumps(report))
    decode = validate_formation if formation else validate_prefix
    decode(root, root/'observations', coupled_extension=True)


def update_coupled_report(root, values):
    update_report(root, values)
    report = json.loads((root/'native_report.json').read_text())
    receipt = root/'observations/recording.json'
    recording = json.loads(receipt.read_text())
    recording['coupled_extension'] = report['coupled_extension']
    receipt.write_text(json.dumps(recording))


class CoupledFormationReplay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original, self.formation = self.root/'coupled-prefix', self.root/'coupled-formation'
        coupled_fixture(self.original)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, name='coupled.json', *, coupled_extension=True, original=None, formation=None):
        return replay.compare(original or self.original, formation or self.formation, self.root/name,
                              workdir=self.root, coupled_extension=coupled_extension)

    def failure(self, name, match, **kwargs):
        with self.assertRaisesRegex(ValueError, match):
            self.compare(name, **kwargs)
        result = json.loads((self.root/name).read_text())
        self.assertFalse(result['scientific_fidelity_qualified'])
        return result

    def test_complete_coupled_population_checkpoint_and_prefix_events(self):
        coupled_fixture(self.formation, self.original)
        with patch.object(replay, 'validate_coupled_identity', wraps=replay.validate_coupled_identity) as validate:
            result = self.compare()
            self.assertEqual(validate.call_count, 2)
        self.assertEqual(result['status'], 'PASS_COUPLED_EXTENSION_FORMATION_PREFIX_REPLAY_AND_COUNTS')
        self.assertEqual(result['units'], 168106)
        self.assertEqual(len(result['groups']), 8)
        self.assertEqual(result['prefix_events_compared'], 18)
        self.assertEqual([row['events'] for row in result['independently_reconstructed_counts']], [18,34])
        self.assertEqual(result['independently_reconstructed_counts'][1]['field_count_values'], 560000)
        self.assertEqual(sum(group['future_records'] for group in result['groups']), 16)
        self.assertTrue(result['checks'][0]['equal'])
        self.assertEqual(result['checks'][0]['ignored_byte_range'], [12,16])
        self.assertTrue(result['prefix_event_multisets_equal'])
        self.assertEqual(result['coupled_extension'], coupled_declaration())
        self.assertFalse(result['full_scale_qualified'])
        self.assertFalse(result['independent_full_formation_execution_replay'])
        self.assertFalse(result['complete_formation_retrieval'])

    def test_explicit_selection_and_both_mixed_family_directions_are_rejected(self):
        coupled_fixture(self.formation, self.original)
        result = self.failure('without-flag.json', 'completed bound native execution required', coupled_extension=False)
        self.assertEqual(result['status'], 'FAILED_NATIVE_FORMATION_REPLAY')
        native_prefix, native_formation = self.root/'native-prefix', self.root/'native-formation'
        original_fixture(native_prefix)
        formation_fixture(native_formation, native_prefix)
        for name, left, right in [('native-left.json',native_prefix,self.formation),
                                  ('native-right.json',self.original,native_formation)]:
            result = self.failure(name, 'CA3_COUPLED_SCOPE_REQUIRED', original=left, formation=right)
            self.assertEqual(result['status'], 'FAILED_COUPLED_EXTENSION_FORMATION_REPLAY')

    def test_shared_model_identity_and_complete_formation_are_required(self):
        coupled_fixture(self.formation, self.original)
        update_coupled_report(self.formation, {'core_neurons':84053})
        result = self.failure('wrong-units.json', 'CA3_COUPLED_MODEL_IDENTITY_DIFFERS')
        self.assertEqual(result['status'], 'FAILED_COUPLED_EXTENSION_FORMATION_REPLAY')
        update_coupled_report(self.formation, {'core_neurons':168106, 'complete_formation_executed':False})
        self.failure('incomplete.json', 'complete coupled extension formation required')
        update_coupled_report(self.formation, {'complete_formation_executed':True,
                                             'complete_author_formation_program_unchanged':True})
        self.failure('false-native-claim.json', 'complete coupled extension formation required')

    def test_full_extension_declaration_and_source_fields_must_match(self):
        coupled_fixture(self.formation, self.original)
        changed = coupled_declaration()
        changed['source_pattern_sha256'] = '7'*64
        update_coupled_report(self.formation, {'coupled_extension':changed})
        self.failure('different-extension.json', 'coupled extension identity changed: coupled_extension')
        update_coupled_report(self.formation, {'coupled_extension':coupled_declaration(), 'pattern_sha256':'6'*64})
        self.failure('different-pattern.json', 'coupled extension identity changed: pattern_sha256')

    def test_checkpoint_and_exact_pairs_keep_derived_mismatch_family(self):
        for name, options in [('checkpoint.json', {'corrupt_prefix':True}),
                               ('pairs.json', {'changed_pairs':True})]:
            if self.formation.exists():
                shutil.rmtree(self.formation)
            coupled_fixture(self.formation, self.original, **options)
            result = self.failure(name, 'formation prefix checkpoint or exact event multiset differs')
            self.assertEqual(result['status'], 'MISMATCH_COUPLED_EXTENSION_FORMATION_PREFIX_REPLAY')
            if options.get('changed_pairs'):
                self.assertTrue(result['checks'][0]['equal'])
                self.assertFalse(result['prefix_event_multisets_equal'])
            else:
                self.assertFalse(result['checks'][0]['equal'])
                self.assertTrue(result['prefix_event_multisets_equal'])

    def test_decoder_must_bind_the_same_derived_identity(self):
        coupled_fixture(self.formation, self.original)
        receipt = self.formation/'observations/recording.json'
        recording = json.loads(receipt.read_text())
        recording['coupled_extension']['derived_assembly_size'] = 275
        receipt.write_text(json.dumps(recording))
        self.failure('decoder-identity.json', 'bound decoder receipt required')

    def test_cli_requires_the_explicit_coupled_extension_switch(self):
        coupled_fixture(self.formation, self.original)
        command = [sys.executable, '-m', 'population_dynamics.carlsim_formation_replay',
                   '--original',str(self.original), '--formation',str(self.formation), '--workdir',str(self.root)]
        env = {**os.environ, 'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        good = subprocess.run([*command, '--coupled-extension', '--output',str(self.root/'cli-coupled.json')],
                              capture_output=True, text=True, env=env)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)['status'], 'PASS_COUPLED_EXTENSION_FORMATION_PREFIX_REPLAY_AND_COUNTS')
        bad = subprocess.run([*command, '--output',str(self.root/'cli-default.json')],
                             capture_output=True, text=True, env=env)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(json.loads((self.root/'cli-default.json').read_text())['status'], 'FAILED_NATIVE_FORMATION_REPLAY')


if __name__ == '__main__':
    unittest.main()
