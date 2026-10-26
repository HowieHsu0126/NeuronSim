import os
from pathlib import Path
import unittest

import numpy as np

from .graph_components import CudaWeakComponents

LIBRARY=os.environ.get('GRAPH_COMPONENTS_LIBRARY')


def reference(n,edges):
    adjacency=[[] for _ in range(n)]
    for a,b in edges: adjacency[a].append(b);adjacency[b].append(a)
    result=np.full(n,-1,dtype=np.int32)
    for i in range(n):
        if result[i]>=0:continue
        stack=[i];result[i]=i
        while stack:
            for j in adjacency[stack.pop()]:
                if result[j]<0:result[j]=i;stack.append(j)
    return result


@unittest.skipUnless(LIBRARY,'GRAPH_COMPONENTS_LIBRARY required; no CPU fallback')
class ComponentTests(unittest.TestCase):
    def test_disconnected_cycles_self_edges_and_isolates(self):
        edges=np.array([[0,1],[1,2],[2,0],[4,5],[5,4],[6,6]],dtype=np.int64)
        with CudaWeakComponents(Path(LIBRARY),8) as c:
            c.add(edges[:,0],edges[:,1])
            np.testing.assert_array_equal(c.labels(),reference(8,edges))
            c.add(np.array([2,3]),np.array([4,5]))
            np.testing.assert_array_equal(c.labels(),reference(8,np.r_[edges,[[2,4],[3,5]]]))

    def test_shuffled_streamed_graphs_match_independent_breadth_first_components(self):
        rng=np.random.default_rng(78001)
        for n in [1,37,2048]:
            for count in [0,n//2,n*5]:
                edges=rng.integers(0,n,(count,2));expected=reference(n,edges)
                with CudaWeakComponents(Path(LIBRARY),n) as c:
                    for batch in np.array_split(edges,17):c.add(batch[:,0],batch[:,1])
                    np.testing.assert_array_equal(c.labels(),expected)
                rng.shuffle(edges)
                with CudaWeakComponents(Path(LIBRARY),n) as c:
                    c.add(edges[:,1],edges[:,0]);np.testing.assert_array_equal(c.labels(),expected)

    def test_invalid_endpoints_and_closed_lifetime(self):
        with CudaWeakComponents(Path(LIBRARY),5) as c:
            for source,target in [([-1],[0]),([5],[0]),([0.5],[1.]),([0,1],[0])]:
                with self.assertRaises(ValueError):c.add(source,target)
        with self.assertRaises(RuntimeError):c.labels()


if __name__=='__main__':unittest.main()
