import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from .carlsim_normalization import normalize
from .carlsim_records import NETWORK_HEADER, GROUP_HEADER, NETWORK_SIGNATURE


def checkpoint(path, rows, *, networks=1):
    with path.open('wb') as stream:
        stream.write(NETWORK_HEADER.pack(NETWORK_SIGNATURE, 0.3, 45.85, 12.0, 5, 2))
        stream.write(GROUP_HEADER.pack(0, 1, 2, 1, 1, b'CA3_I'))
        stream.write(GROUP_HEADER.pack(2, 4, 3, 1, 1, b'CA3_Pyramidal'))
        stream.write(struct.pack('<ii', networks, len(rows)))
        for row in rows:
            stream.write(struct.pack('<iiiiiffi', *row))


class Normalization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'input.dat'
        self.rows = [(1,1,0,1,50,0.03125,2.0,1), (0,1,0,2,7,-0.9,4.0,1),
                     (1,1,2,0,50,1.0000001192092896,3.0,1), (1,1,1,1,50,25.25,8.0,1)]
        checkpoint(self.source, self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def run_normalizer(self, name='normalized', chunk=2):
        return normalize(self.source,self.root/(name+'.dat'),self.root/(name+'.json'),
                         self.root/'scratch',chunk_records=chunk)

    def test_global_float64_mean_and_only_selected_weight_bytes_change(self):
        before=self.source.read_bytes()
        report=self.run_normalizer()
        offset=NETWORK_HEADER.size+2*GROUP_HEADER.size+8
        actual=(self.root/'normalized.dat').read_bytes()
        expected=bytearray(before)
        weights=[struct.unpack_from('<f',before,offset+i*32+20)[0]
                 for i,row in enumerate(self.rows) if row[4]==50]
        mean=np.mean(weights)
        for i,row in enumerate(self.rows):
            if row[4]==50:
                value=struct.unpack_from('<f',before,offset+i*32+20)[0]
                struct.pack_into('<f',expected,offset+i*32+20,value/mean*0.625)
        self.assertEqual(actual,bytes(expected))
        self.assertEqual(self.source.read_bytes(),before)
        self.assertEqual(report['selected_synapses'],3)
        self.assertFalse(report['retrieval_executed'])
        self.assertEqual(list((self.root/'scratch').iterdir()),[])
        self.run_normalizer('one_record',chunk=1)
        self.assertEqual(actual,(self.root/'one_record.dat').read_bytes())

    def test_zero_mean_and_wrong_connection_mapping_are_rejected(self):
        checkpoint(self.source,[(1,1,0,1,50,0.0,2.0,1)])
        with self.assertRaisesRegex(ValueError,'POSITIVE_GLOBAL'):
            self.run_normalizer()
        self.assertEqual(json.loads((self.root/'normalized.json').read_text())['status'],'FAILED')
        self.assertFalse((self.root/'normalized.dat').exists())
        checkpoint(self.source,[(0,1,0,1,50,1.0,2.0,1)])
        with self.assertRaisesRegex(ValueError,'NOT_PYRAMIDAL'):
            self.run_normalizer('wrong_mapping')

    def test_payload_extent_block_count_and_unit_identity_are_checked(self):
        self.source.write_bytes(self.source.read_bytes()+b'extra')
        with self.assertRaisesRegex(ValueError,'LENGTH_MISMATCH'):
            self.run_normalizer()
        checkpoint(self.source,self.rows,networks=2)
        with self.assertRaisesRegex(ValueError,'ONE_NONEMPTY_NETWORK'):
            self.run_normalizer()
        checkpoint(self.source,[(1,1,3,1,50,1.0,2.0,1)])
        with self.assertRaisesRegex(ValueError,'GROUP_LOCAL_UNIT'):
            self.run_normalizer()

    def test_empty_selected_set_and_existing_outputs_fail(self):
        checkpoint(self.source,[self.rows[1]])
        with self.assertRaisesRegex(ValueError,'EMPTY_PYRAMIDAL'):
            self.run_normalizer()
        checkpoint(self.source,self.rows)
        with self.assertRaises(FileExistsError):
            self.run_normalizer()


if __name__ == '__main__':
    unittest.main()
