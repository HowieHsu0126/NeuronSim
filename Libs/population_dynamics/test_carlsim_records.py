from pathlib import Path
import json
import struct
import tempfile
import unittest

import numpy as np

from .carlsim_records import (GROUP_HEADER, NETWORK_HEADER, NETWORK_SIGNATURE,
    SPIKE_HEADER, SPIKE_SIGNATURE, network_header, spike_blocks, validate_prefix, _sha)


def network(path, *, seconds=4.45, groups=None):
    groups=groups or [(0,1,2,1,1,b'I'),(2,4,3,1,1,b'E')]
    payload=NETWORK_HEADER.pack(NETWORK_SIGNATURE,0.3,seconds,1.0,5,len(groups))
    path.write_bytes(payload+b''.join(GROUP_HEADER.pack(*g) for g in groups))


def events(path, units, rows):
    path.write_bytes(SPIKE_HEADER.pack(SPIKE_SIGNATURE,0.2,units,1,1)+
                     b''.join(struct.pack('<ii',*r) for r in rows))


class SourceRecords(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_header_global_ids_not_group_order_and_no_restart_claim(self):
        p=self.root/'network.dat';network(p,groups=[(2,4,3,1,1,b'E'),(0,1,2,1,1,b'I')])
        result=network_header(p)
        self.assertEqual([(g.name,g.start,g.stop) for g in result['groups']],[('E',2,5),('I',0,2)])
        self.assertEqual(result['payload_offset'],264)
        self.assertFalse(result['full_state_restart_available'])

    def test_event_order_duplicates_and_chunk_boundaries_are_preserved(self):
        p=self.root/'spikes.dat';rows=[(7,1),(7,1),(1,0),(999,2),(500,1)]
        events(p,3,rows)
        for chunk in (1,2,3,10):
            actual=np.concatenate(list(spike_blocks(p,units=3,stop_ms=1000,chunk_records=chunk)))
            np.testing.assert_array_equal(actual,np.array(rows))

    def test_event_time_id_and_truncation_rejected(self):
        p=self.root/'spikes.dat'
        for rows in [[(-1,0)],[(1000,0)],[(4,3)],[(0,-1)]]:
            events(p,3,rows)
            with self.assertRaisesRegex(ValueError,'OUT_OF_RANGE'):
                list(spike_blocks(p,units=3,stop_ms=1000))
        events(p,3,[(0,1)]);p.write_bytes(p.read_bytes()+b'X')
        with self.assertRaisesRegex(ValueError,'TRUNCATED_EVENT'):
            list(spike_blocks(p,units=3,stop_ms=1000))
        p.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'TRUNCATED_HEADER'):
            list(spike_blocks(p,units=3,stop_ms=1000))

    def test_unknown_versions_and_unit_coverage_rejected(self):
        p=self.root/'spikes.dat';p.write_bytes(SPIKE_HEADER.pack(SPIKE_SIGNATURE,0.3,3,1,1))
        with self.assertRaisesRegex(ValueError,'FORMAT_UNSUPPORTED'):
            list(spike_blocks(p,units=3,stop_ms=1000))
        events(p,4,[])
        with self.assertRaisesRegex(ValueError,'UNIT_COUNT_MISMATCH'):
            list(spike_blocks(p,units=3,stop_ms=1000))
        q=self.root/'network.dat';network(q,groups=[(0,1,2,1,1,b'I'),(1,3,3,1,1,b'E')])
        with self.assertRaisesRegex(ValueError,'COVERAGE_INVALID'):
            network_header(q)

    def test_complete_reader_keeps_global_identity_and_half_open_ms(self):
        run=self.root/'native_run';(run/'results').mkdir(parents=True)
        checkpoint=run/'network5p.dat';network(checkpoint)
        events(run/'results/spk_I.dat',2,[(0,0),(12,1)])
        events(run/'results/spk_E.dat',3,[(20,2),(20,2),(2,0),(4449,1)])
        report=dict(status='NATIVE_PREFIX_EXIT_ZERO',exit_code=0,
            framework_revision='c4955998b950b9c561b3373a71b611aedc705be9',
            scope='native_source_prefix',native_run_executed=True,
            checkpoint_sha256=_sha(checkpoint),expected_prefix_ms=4450,core_neurons=5,
            populations={'I':2,'E':3})
        (self.root/'native_report.json').write_text(json.dumps(report))
        output=self.root/'observations';result=validate_prefix(self.root,output)
        self.assertEqual(result['records'],6)
        with np.load(output/'population_counts.npz') as data:
            np.testing.assert_array_equal(data['unit_counts'],[1,1,1,1,2])
            self.assertEqual(data['population_counts_per_ms'][20,1],2)
            self.assertEqual(data['population_counts_per_ms'][4449,1],1)
        self.assertFalse(result['independent_native_spike_counter_verified'])
        self.assertEqual(result['production_episodes'],0)
        with self.assertRaises(FileExistsError):
            validate_prefix(self.root,output)
        report['expected_prefix_ms']=4451
        (self.root/'native_report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'TIME_MISMATCH'):
            validate_prefix(self.root,self.root/'other')
        self.assertFalse((self.root/'other').exists())


if __name__=='__main__':
    unittest.main()
