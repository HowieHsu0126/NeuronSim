"""Contract tests for source-preserving MPI snapshot replay."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .nest_gpu_mpi_network import background_program, compare, load_drive, partition, unique_source_batches
from .provenance import sha256_file


class MpiSourceMappingTests(unittest.TestCase):
    def test_unique_calls_retain_entire_multigraph_and_attributes(self):
        dtype=[('source','i8'),('target','i8'),('weight','f8'),('delay','f8')]
        rows=np.array([(9,2,1.,.3),(4,7,-2.,.4),(9,2,3.,.5),(9,5,1.,.3),(4,7,-2.,.4)],dtype=dtype)
        groups=list(unique_source_batches(rows))
        self.assertEqual(len(groups),3)
        for group in groups:self.assertEqual(len(np.unique(group['source'])),len(group))
        self.assertEqual(sorted(np.concatenate(groups).tolist()),sorted(rows.tolist()))
        self.assertEqual(list(unique_source_batches(rows[:0])),[])

    def test_spatial_partition_preserves_ids_and_local_group_order(self):
        arrays={'unit_ids':np.array([7,9,13,20]),'positions_mm':np.array([[-1.,0],[1.,0],[0.,1],[-.2,0]])}
        plan={'cortex':np.array([7,9]),'tc':np.array([13,20])}
        owners,maps,groups=partition(arrays,plan,True)
        self.assertEqual(owners,{7:0,9:1,13:1,20:0})
        self.assertEqual(maps,[{7:0,20:1},{9:0,13:1}])
        np.testing.assert_array_equal(np.sort(np.concatenate([np.concatenate(g) for g in groups])),arrays['unit_ids'])
        arrays['positions_mm'][:,0]=1
        with self.assertRaisesRegex(ValueError,'each partition'):partition(arrays,plan,True)

    def test_fixed_background_multiplicity_and_source_clock(self):
        edges=np.array([(2,1,.2),(1,1,.1)],dtype=[('target','i8'),('weight','f8'),('delay','f8')])
        counts=np.zeros((8,2),dtype=np.int64);counts[3,0]=3;counts[6,1]=2
        programs=background_program({'counts':counts},edges,np.array([1,2]),.1)
        np.testing.assert_array_equal(programs[1]['emission_ticks'],[2])
        np.testing.assert_array_equal(programs[2]['emission_ticks'],[4])
        np.testing.assert_array_equal(programs[1]['multiplicities'],[3])
        counts[1,0]=1
        with self.assertRaisesRegex(ValueError,'strictly positive'):background_program({'counts':counts},edges,np.array([1,2]),.1)

    def test_background_requires_verified_identity_and_valid_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);snapshot=root/'snapshot';snapshot.mkdir();(snapshot/'manifest.json').write_text('{}')
            directory=root/'drive';directory.mkdir();counts=np.zeros((4,2),dtype=np.int64)
            def write():
                np.savez(directory/'counts.npz',unit_ids=np.array([1,2]),arrival_ticks=np.arange(4),counts=counts,dt_ms=.1)
                receipt=dict(schema='mesocircuit_background_arrivals_v1',status='PASS',counts_decoding_validated=True,
                    spike_capture_complete_against_native_counters=True,snapshot_manifest_sha256=sha256_file(snapshot/'manifest.json'),
                    counts_sha256=sha256_file(directory/'counts.npz'))
                (directory/'drive.json').write_text(json.dumps(receipt))
            write();meta={'duration_ms':.4,'dt_ms':.1};plan={'cortex':np.array([1,2])}
            self.assertEqual(load_drive(directory,snapshot,meta,plan)['counts'].shape,(4,2))
            with self.assertRaisesRegex(ValueError,'identity'):load_drive(directory,snapshot,meta,{'cortex':np.array([2,1])})
            counts[0,0]=-1;write()
            with self.assertRaisesRegex(ValueError,'count shape'):load_drive(directory,snapshot,meta,plan)
            counts[0,0]=0;write();(snapshot/'manifest.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'verified'):load_drive(directory,snapshot,meta,plan)


    def test_comparison_rejects_omission_and_unverified_native_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);reference=root/'reference';distributed=root/'distributed'
            reference.mkdir();distributed.mkdir()
            for parent,rank,ids in [(reference,0,[1,2]),(distributed,0,[1]),(distributed,1,[2])]:
                filename=parent/f'rank_{rank}.npz'
                np.savez(filename,unit_ids=np.array(ids),recorded_unit_ids=np.array(ids),event_ids=np.array([],dtype=int),
                    event_ticks=np.array([],dtype=int),ticks=np.array([0,1]),states=np.zeros((2,len(ids),4)))
                is_distributed=parent==distributed
                report=dict(status='EXECUTED',arrays_sha256=sha256_file(filename),hosts=2 if is_distributed else 1,
                    rank=rank,distributed=is_distributed,omit_cross=False,incoming_edges=len(ids),
                    expected_in_window_cortical_cross_arrivals=1 if is_distributed else 0,
                    checks={'native_incoming_connection_count_exact':True},snapshot_manifest_sha256='same-snapshot',
                    drive_report_sha256='same-drive',schedule='effective',native_library_sha256='same-library',
                    cuda_visible_devices=str(rank) if is_distributed else '0,1')
                (parent/f'rank_{rank}.json').write_text(json.dumps(report))
            path=distributed/'rank_1.json';report=json.loads(path.read_text());report['omit_cross']=True;path.write_text(json.dumps(report))
            result=compare(reference,distributed,root/'comparison.json')
            self.assertEqual(result['status'],'FAILED');self.assertFalse(result['checks']['coupled_conditions'])
            report['omit_cross']=False;report['checks']['native_incoming_connection_count_exact']=False;path.write_text(json.dumps(report))
            result=compare(reference,distributed,root/'comparison.json')
            self.assertEqual(result['status'],'FAILED');self.assertFalse(result['checks']['native_edge_counts_verified'])

if __name__=='__main__':unittest.main()
