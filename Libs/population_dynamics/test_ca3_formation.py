import json
from pathlib import Path
import struct
import tempfile
import unittest

from .ca3_reference import formation_program, prepare_formation_program, verify_checkpoints
from .carlsim_records import _sha, validate_formation, validate_prefix
from .test_carlsim_records import network, events


class Formation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def saved_network(self, path, seconds):
        network(path)
        payload = bytearray(path.read_bytes())
        payload[8:12] = struct.pack('<f', seconds)
        path.write_bytes(payload)

    def test_explicit_full_program_preserves_every_source_byte(self):
        source = '// source\nsim.saveSimulation("network5p.dat", true);\nnext_cycle();\n'
        self.assertEqual(formation_program(source, 'full-formation'), source)
        prefix = formation_program(source, 'prefix')
        self.assertEqual(prefix.replace('\n                return 0; // declared source-prefix stop; destructors flush native monitors\n', ''), source)
        for bad in ['full', '', 'retrieval']:
            with self.assertRaisesRegex(ValueError, 'SCOPE'):
                formation_program(source, bad)
        with self.assertRaisesRegex(ValueError, 'BOUNDARY'):
            formation_program('changed source', 'full-formation')

    def test_full_file_keeps_source_crlf_while_prefix_retains_legacy_overlay(self):
        raw = b'// source\r\nsim.saveSimulation("network5p.dat", true);\r\nnext_cycle();\r\n'
        path = self.root/'main.cpp'
        path.write_bytes(raw)
        prepare_formation_program(path, 'full-formation')
        self.assertEqual(path.read_bytes(), raw)
        legacy_input = path.read_text()
        prepare_formation_program(path, 'prefix')
        self.assertEqual(path.read_text(), formation_program(legacy_input, 'prefix'))

    def test_all_twenty_checkpoints_and_source_clock_required(self):
        for n in range(5, 101, 5):
            self.saved_network(self.root/f'network{n}p.dat', (1000 + n*690)/1000)
        rows = verify_checkpoints(self.root, 'full-formation', expected_units=5)
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]['simulation_ms'], 4450)
        self.assertEqual(rows[12]['simulation_ms'], 45850)
        self.assertEqual(rows[-1]['simulation_ms'], 70000)
        self.assertEqual(len(verify_checkpoints(self.root, 'prefix', expected_units=5)), 1)
        self.saved_network(self.root/'network65p.dat', 45.0)
        with self.assertRaisesRegex(ValueError, 'TIME_DIFFERS'):
            verify_checkpoints(self.root, 'full-formation', expected_units=5)
        with self.assertRaisesRegex(ValueError, 'UNIT_IDENTITY'):
            verify_checkpoints(self.root, 'prefix', expected_units=6)

    def test_full_native_decoder_preserves_population_ids_and_half_open_duration(self):
        run = self.root/'native_run'
        (run/'results').mkdir(parents=True)
        checkpoint = run/'network100p.dat'
        self.saved_network(checkpoint, 70.0)
        events(run/'results/spk_I.dat', 2, [(0, 0), (69999, 1)])
        events(run/'results/spk_E.dat', 3, [(1, 0), (60000, 2)])
        report = dict(status='NATIVE_FORMATION_EXIT_ZERO', exit_code=0,
                      framework_revision='c4955998b950b9c561b3373a71b611aedc705be9',
                      scope='native_source_formation', native_run_executed=True,
                      expected_duration_ms=70000, complete_formation_executed=True,
                      checkpoint_sha256=_sha(checkpoint), core_neurons=5, populations={'I': 2, 'E': 3})
        path = self.root/'native_report.json'
        path.write_text(json.dumps(report))
        result = validate_formation(self.root, self.root/'observation')
        self.assertEqual(result['status'], 'DECODED_NATIVE_FORMATION_FILES')
        self.assertEqual(result['records'], 4)
        self.assertEqual(result['duration_ms'], 70000)
        self.assertTrue(result['complete_formation_executed'])
        self.assertFalse(result['complete_formation_retrieval'])
        self.assertFalse(result['scientific_fidelity_qualified'])
        self.assertEqual(result['production_episodes'], 0)
        with self.assertRaisesRegex(ValueError, 'PINNED_PREFIX'):
            validate_prefix(self.root, self.root/'wrong_scope')
        events(run/'results/spk_E.dat', 3, [(70000, 0)])
        with self.assertRaisesRegex(ValueError, 'OUT_OF_RANGE'):
            validate_formation(self.root, self.root/'wrong_boundary')
        report['complete_formation_executed'] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'COMPLETE_FORMATION'):
            validate_formation(self.root, self.root/'incomplete')


if __name__ == '__main__':
    unittest.main()
