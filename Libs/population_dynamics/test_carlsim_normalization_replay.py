import json
from pathlib import Path
import struct
import tempfile
import unittest

from .carlsim_normalization import normalize
from .carlsim_normalization_replay import verify
from .provenance import sha256_file
from .test_carlsim_normalization import checkpoint


class FullNormalizationReplay(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/'source.dat'
        self.target = self.root/'normalized.dat'
        self.report = self.root/'normalization.json'
        self.rows = [(1,1,0,1,50,0.03125,2.0,1), (0,1,0,2,7,-0.0,4.0,1),
                     (1,1,2,0,50,1.0000001192092896,3.0,1), (1,1,1,1,50,25.25,8.0,1)]
        checkpoint(self.source, self.rows)
        normalize(self.source, self.target, self.report, self.root/'scratch', chunk_records=1)

    def tearDown(self):
        self.temporary.cleanup()

    def replay(self):
        return verify(self.source, self.target, self.report, self.root/'replay.json',
                      self.root/'scratch', chunk_records=3)

    def rebind_target(self):
        value = json.loads(self.report.read_text())
        value['output_sha256'] = sha256_file(self.target)
        self.report.write_text(json.dumps(value))

    def test_all_records_and_header_pass_exactly(self):
        result = self.replay()
        self.assertEqual(result['status'], 'PASS_NORMALIZATION_REPLAY')
        self.assertEqual(result['synapses'], 4)
        self.assertEqual(result['selected_synapses'], 3)
        self.assertEqual(result['bytes_compared'], self.target.stat().st_size)
        self.assertFalse(result['scientific_fidelity_qualified'])
        self.assertEqual(list((self.root/'scratch').iterdir()), [])
        with self.assertRaises(FileExistsError):
            self.replay()

    def test_nonselected_signed_zero_corruption_is_detected_even_with_rebound_hash(self):
        payload = bytearray(self.target.read_bytes())
        struct.pack_into('<f', payload, 272+32+20, 0.0)
        self.target.write_bytes(payload)
        self.rebind_target()
        with self.assertRaisesRegex(ValueError, 'record1, word5'):
            self.replay()
        self.assertEqual(json.loads((self.root/'replay.json').read_text())['status'],
                         'FAILED_NORMALIZATION_REPLAY')

    def test_selected_weight_or_nonweight_changes_fail(self):
        before = self.target.read_bytes()
        for index in (20, 24, 28):
            with self.subTest(offset=index):
                payload = bytearray(before)
                payload[272+index] ^= 1
                self.target.write_bytes(payload)
                self.rebind_target()
                with self.assertRaisesRegex(ValueError, 'normalized bytes differ'):
                    self.replay()
                (self.root/'replay.json').unlink()

    def test_header_and_mean_tampering_are_rejected(self):
        original = self.target.read_bytes()
        payload = bytearray(original)
        payload[12] ^= 1
        self.target.write_bytes(payload)
        self.rebind_target()
        with self.assertRaisesRegex(ValueError, 'header'):
            self.replay()
        (self.root/'replay.json').unlink()
        self.target.write_bytes(original)
        self.rebind_target()
        report = json.loads(self.report.read_text())
        report['original_global_mean'] += 0.1
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'counts or mean'):
            self.replay()


if __name__ == '__main__':
    unittest.main()
