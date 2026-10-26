import unittest
import numpy as np
from unittest.mock import patch
from Libs.population_dynamics.nest_gpu_connectivity import aligned_batches, ExplicitConnector


class ConnectorTests(unittest.TestCase):
    def test_boundary_split_uses_global_offset(self):
        self.assertEqual(list(aligned_batches(14, 7, 16, 100)), [(0, 2), (2, 7)])
        self.assertEqual(list(aligned_batches(16, 17, 16, 100)), [(0, 16), (16, 17)])

    def test_external_batch_bound_is_also_respected(self):
        self.assertEqual(list(aligned_batches(14, 7, 16, 3)), [(0, 2), (2, 5), (5, 7)])

    def test_invalid_counter_is_rejected(self):
        for start in (-1, .5, True):
            with self.assertRaises(ValueError):
                list(aligned_batches(start, 1))

    def test_signed_multapses_remain_distinct(self):
        class Backend:
            def __init__(self): self.rows = []
            def Connect(self, source, target, rule, syn):
                self.rows.extend(zip(source, target, list(syn["weight_array"]),
                                     list(syn["delay_array"]), [syn["receptor"]] * len(source)))
        backend = Backend()
        connector = ExplicitConnector(backend, 0, 2)
        receipt = connector.connect([0, 0, 1], [2, 2, 3], [1., -2., 4.],
                                    [.5, .25, 1.], [0, 1, 0])
        self.assertEqual(sorted(backend.rows),
                         [(0, 2, -2., .25, 1), (0, 2, 1., .5, 0), (1, 3, 4., 1., 0)])
        self.assertEqual(receipt["edges_added"], 3)

    def test_invalid_edges_fail_before_native_creation(self):
        class Backend:
            def Connect(self, *args): raise AssertionError("must not call native backend")
        with self.assertRaises(ValueError):
            ExplicitConnector(Backend()).connect([0], [1], [float("nan")], [.1], [0])
        with self.assertRaises(ValueError):
            ExplicitConnector(Backend()).connect([0], [], [1.], [.1], [0])


class ArrayConnectorTests(unittest.TestCase):
    def test_array_path_retains_signed_multiedges_and_storage_boundaries(self):
        rows=[]
        def native(ng,s,t,w,d,p):
            rows.append(list(zip(s.tolist(),t.tolist(),w.tolist(),d.tolist(),[p]*len(s))))
        with patch("Libs.population_dynamics.nest_gpu_connectivity.raw_connect_arrays",native):
            from .nest_gpu_connectivity import NATIVE_CONNECTION_BLOCK_SIZE
            connector=ExplicitConnector(None,NATIVE_CONNECTION_BLOCK_SIZE-1,2)
            receipt=connector.connect_arrays(np.array([0,0,1,1]),np.array([2,2,3,3]),
                np.array([1.,-2.,4.,5.]),np.array([.5,.25,1.,2.]),np.array([0,1,0,0]))
        self.assertEqual([len(x) for x in rows],[1,2,1])
        self.assertEqual(sorted(x for r in rows for x in r),
            [(0,2,-2.,.25,1),(0,2,1.,.5,0),(1,3,4.,1.,0),(1,3,5.,2.,0)])
        self.assertEqual(receipt["edges_added"],4)

    def test_array_path_rejects_before_native_int_or_float_overflow(self):
        for source,weight in [(np.array([2**31]),np.array([1.])),
                              (np.array([1.]),np.array([1.])),
                              (np.array([1]),np.array([1e100]))]:
            with self.assertRaises(ValueError):ExplicitConnector(None).connect_arrays(
                source,np.array([2]),weight,np.array([.1]),np.array([0]))


if __name__ == "__main__":
    unittest.main()
