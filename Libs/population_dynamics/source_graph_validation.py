"""Validate streamed source graph identities, attributes and weak connectivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .graph_components import CudaWeakComponents
from .provenance import sha256_file
from .spatial_connectivity import EDGE_DTYPE,SCHEMA,native_delay_ticks


def validate(graph: Path, library: Path, output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report={'status':'FAILED','scientific_reference_qualified':False,'dynamics_simulated':False}
    try:
        m=json.loads((graph/'manifest.json').read_text())
        if m.get('schema')!=SCHEMA or m.get('status')!='GENERATED':
            raise ValueError('complete source graph schema required')
        if sha256_file(graph/'layout.npz')!=m['layout_sha256']:
            raise ValueError('source layout checksum mismatch')
        with np.load(graph/'layout.npz',allow_pickle=False) as layout:
            ids,labels=layout['unit_ids'],layout['populations']
        if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or labels.shape!=ids.shape:
            raise ValueError('unique labeled source units required')
        order=np.argsort(ids);sorted_ids=ids[order]
        contiguous=np.array_equal(ids,np.arange(1,len(ids)+1))
        def indices(values):
            if contiguous:
                if np.any(values<1) or np.any(values>len(ids)): raise ValueError('unknown source unit')
                return values-1
            take=np.searchsorted(sorted_ids,values)
            if np.any(take==len(ids)) or not np.array_equal(sorted_ids[take],values):
                raise ValueError('unknown source unit')
            return order[take]
        names=m['resolved_source_network']['populations']
        required={(t,s) for t in names[:-1] for s in names}
        projections={(p['target_population'],p['source_population']):p for p in m['projections']}
        if len(m['projections'])!=72 or set(projections)!=required:
            raise ValueError('complete unique source projections required')
        if len({p['path'] for p in m['parts']})!=len(m['parts']):
            raise ValueError('duplicate graph shard declaration')
        dt=float(m['source_config']['simulation']['sim_resolution'])
        indegree=np.zeros(len(ids),dtype=np.int64);outdegree=np.zeros(len(ids),dtype=np.int64)
        observed={key:0 for key in projections};total=0
        with CudaWeakComponents(library,int(len(ids))) as components:
            for part in m['parts']:
                name=part['path']
                if Path(name).name!=name:raise ValueError('graph shard must be a local filename')
                file=graph/name
                if sha256_file(file)!=part['sha256']:raise ValueError('graph shard checksum mismatch: '+name)
                edges=np.load(file,mmap_mode='r',allow_pickle=False)
                if edges.dtype!=EDGE_DTYPE or edges.ndim!=1 or len(edges)!=part['edges']:
                    raise ValueError('graph shard shape or dtype mismatch')
                key=part['target_population'],part['source_population']
                p=projections[key]['parameters']
                source,target=indices(edges['source']),indices(edges['target'])
                if (not np.all(labels[source]==key[1]) or not np.all(labels[target]==key[0])
                        or np.any(source==target)):
                    raise ValueError('source/target population or autapse violation')
                if (np.any(edges['repeat']<0) or np.any(edges['repeat']>=p['repeats'])
                        or np.any(edges['anchor_index']<0) or np.any(edges['anchor_index']>3)):
                    raise ValueError('invalid source repeat or periodic anchor')
                weight=edges['weight']
                if not np.isfinite(weight).all() or np.any((weight>0) if p['weight_mean']<0 else (weight<0)):
                    raise ValueError('source weight sign or finiteness violation')
                if np.any(edges['raw_delay']<dt/2):raise ValueError('delay outside source redraw support')
                native_delay_ticks(edges['raw_delay'],dt)
                components.add(source,target)
                indegree+=np.bincount(target,minlength=len(ids))
                outdegree+=np.bincount(source,minlength=len(ids))
                observed[key]+=len(edges);total+=len(edges)
            component_labels=components.labels()
        if total!=m['edges'] or any(observed[k]!=projections[k]['edges'] for k in observed):
            raise ValueError('source projection denominator mismatch')
        roots,counts=np.unique(component_labels,return_counts=True)
        cortex=labels!='TC'
        np.savez(output/'connectivity.npz',unit_ids=ids,populations=labels,component_labels=component_labels,
                 indegree=indegree,outdegree=outdegree)
        report.update(status='PASS',units=len(ids),cortical_units=int(cortex.sum()),tc_units=int((~cortex).sum()),
            edges=total,shards=len(m['parts']),projections=len(observed),weak_components=len(roots),
            component_sizes=counts.tolist(),isolated_cortical_units=int(np.sum((indegree[cortex]+outdegree[cortex])==0)),
            minimum_cortical_indegree=int(indegree[cortex].min()),minimum_cortical_outdegree=int(outdegree[cortex].min()),
            graph_manifest_sha256=sha256_file(graph/'manifest.json'),library_sha256=sha256_file(library),
            validation_implementation_sha256=sha256_file(Path(__file__)))
    except Exception as exc:
        report.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report['seconds']=time.perf_counter()-started
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=validate(args.graph,args.library,args.output)
    print(json.dumps(report,indent=2))
    raise SystemExit(report['status']!='PASS')


if __name__=='__main__':main()
