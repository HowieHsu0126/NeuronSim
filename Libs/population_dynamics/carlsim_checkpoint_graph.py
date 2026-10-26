"""Measure the complete nonzero synapse graph in a CARLsim checkpoint.

Uses the existing CA3 binary-layout validator and the same Boolean adjacency
and SciPy component definitions used for V1. Source-index halves are diagnostic
partitions, never anatomical hemispheres. Topology does not qualify dynamics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .carlsim_normalization import SYNAPSE_DTYPE, _layout, _selected
from .carlsim_records import CA3_REFERENCE_POPULATIONS
from .provenance import sha256_file

CHUNK_RECORDS = 65536


def analyze(source, output, *, expected_sha256=None, coupled_extension=False, chunk_records=CHUNK_RECORDS):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(chunk_records) is not int or chunk_records <= 0:
        raise ValueError('CARLSIM_GRAPH_POSITIVE_CHUNK_REQUIRED')
    header, offset, count = _layout(source)
    groups = header['groups']; n = header['units']; group_count = len(groups)
    if coupled_extension and (n != 168106 or {g.name:g.units for g in groups} != {k:2*v for k,v in CA3_REFERENCE_POPULATIONS.items()}):
        raise ValueError('CARLSIM_GRAPH_COUPLED_SOURCE_IDENTITY_REQUIRED')
    matches = [g.group_id for g in groups if g.name == 'CA3_Pyramidal']
    if len(matches) != 1:
        raise ValueError('CARLSIM_GRAPH_PYRAMIDAL_GROUP_REQUIRED')
    starts = np.asarray([g.start for g in groups], dtype=np.int64)
    half_sizes = np.asarray([g.units//2 for g in groups], dtype=np.int64)
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status='SCANNING', source=str(source.resolve()), units=n, synapse_records=count,
        simulation_seconds=header['simulation_seconds'], groups=[asdict(g) for g in groups],
        graph_definition='Boolean directed arcs for each stored weight!=0 with distinct global endpoints; signs and duplicate weights are not summed',
        half_definition='group-local unit index >= floor(group unit count/2); not anatomical position',
        coupled_extension=coupled_extension, numpy_version=np.__version__, scipy_version=scipy.__version__,
        endpoint_allocation_bytes=8*count, scientific_fidelity_qualified=False, source_scale_qualified=False,
        production_episodes=0, native_simulation_executed=False)
    try:
        pre = np.empty(count, dtype=np.int32); post = np.empty(count, dtype=np.int32)
        group_edges = np.zeros((group_count,group_count), dtype=np.int64)
        half_edges = np.zeros((2,2), dtype=np.int64)
        cursor, zero, self_loops, negative = 0,0,0,0
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            prefix = stream.read(offset)
            if len(prefix) != offset:
                raise ValueError('CARLSIM_GRAPH_TRUNCATED_PREFIX')
            digest.update(prefix)
            for first in range(0,count,chunk_records):
                length = min(chunk_records,count-first)
                raw = stream.read(length*SYNAPSE_DTYPE.itemsize)
                if len(raw) != length*SYNAPSE_DTYPE.itemsize:
                    raise ValueError('CARLSIM_GRAPH_TRUNCATED_RECORDS')
                digest.update(raw)
                block = np.frombuffer(raw,dtype=SYNAPSE_DTYPE)
                # Validation covers every record, not only normalization's connection50.
                _selected(block, groups, matches[0])
                pg, qg = block['pre_group'], block['post_group']
                a = starts[pg]+block['pre_unit']; b = starts[qg]+block['post_unit']
                nonzero = block['weight'] != 0
                active = nonzero & (a != b)
                zero += int((~nonzero).sum()); self_loops += int((a == b).sum())
                negative += int((block['weight'] < 0).sum())
                length = int(active.sum())
                pre[cursor:cursor+length] = a[active]; post[cursor:cursor+length] = b[active]
                cursor += length
                group_edges += np.bincount((pg[active]*group_count+qg[active]).astype(np.int64),
                                            minlength=group_count*group_count).reshape(group_count,group_count)
                ah = block['pre_unit'][active] >= half_sizes[pg[active]]
                bh = block['post_unit'][active] >= half_sizes[qg[active]]
                half_edges += np.bincount(ah.astype(np.int64)*2+bh, minlength=4).reshape(2,2)
            if stream.read(1):
                raise ValueError('CARLSIM_GRAPH_TRAILING_BYTES')
        actual_sha = digest.hexdigest()
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError('CARLSIM_GRAPH_SOURCE_HASH_CHANGED')
        report.update(status='MEASURING_COMPONENTS', source_sha256=actual_sha,
            source_hash_matches_declared=expected_sha256 is not None, nonzero_interunit_synapses=cursor,
            zero_weight_records=zero, self_loop_records=self_loops, negative_weight_records=negative,
            nonzero_interunit_group_matrix=group_edges.tolist(), nonzero_interunit_half_matrix=half_edges.tolist())
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        # Boolean COO preserves the presence of parallel signed synapses, even
        # when numeric weights cancel or multiplicity exceeds an integer byte.
        adjacency = coo_matrix((np.ones(cursor,dtype=bool),(pre[:cursor],post[:cursor])),shape=(n,n)).tocsr()
        del pre, post
        labels_saved = dict(unit_ids=np.arange(n,dtype=np.int64))
        components = {}
        for kind in ('strong','weak'):
            total, labels = connected_components(adjacency,directed=True,connection=kind)
            sizes = np.bincount(labels,minlength=total)
            components[kind] = dict(components=int(total),largest_neurons=int(sizes.max(initial=0)),sizes=sizes.tolist())
            labels_saved[kind+'_component'] = labels
        outgoing = np.diff(adjacency.indptr)>0
        incoming = np.bincount(adjacency.indices,minlength=n)>0
        np.savez_compressed(output/'unit_components.npz',**labels_saved)
        report.update(status='MEASURED_CARLSIM_CHECKPOINT_GRAPH',components=components,
            distinct_directed_interunit_pairs=int(adjacency.nnz),
            no_interunit_input_neurons=int((~incoming).sum()),no_interunit_output_neurons=int((~outgoing).sum()),
            isolated_neurons=int((~incoming & ~outgoing).sum()),
            all_units_in_one_strong_component=components['strong']['components']==1,
            all_units_in_one_weak_component=components['weak']['components']==1,
            artifacts={'unit_components.npz':sha256_file(output/'unit_components.npz')},
            implementation_sha256=sha256_file(Path(__file__)))
    except Exception as error:
        report.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--coupled-extension',action='store_true')
    parser.add_argument('--chunk-records',type=int,default=CHUNK_RECORDS)
    args=parser.parse_args(argv)
    print(json.dumps(analyze(args.source,args.output,expected_sha256=args.expected_sha256,
                            coupled_extension=args.coupled_extension,chunk_records=args.chunk_records),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
