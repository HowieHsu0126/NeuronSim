"""Sparse-event observation must equal the existing dense-count mapping."""
import unittest
import numpy as np
from .spatial_field import PeriodicPopulationGrid


class EventFieldTests(unittest.TestCase):
    def grid(self):
        return PeriodicPopulationGrid.from_positions([9,2,7,3],["E","I","E","I"],
            [[-.5,-.5],[.5,.5],[.2,.2],[-.2,.2]],extent_mm=1.,cells_per_axis=2,definition="test")

    def test_matches_dense_projection_and_half_open_boundaries(self):
        grid=self.grid();ids=grid.mapping.unit_ids
        senders=np.array([9,9,2,7,3,3,7,2]);times=np.array([0,1,2,3,4,5,6,7])
        edges=np.array([0,2,4,6]);dense=np.zeros((3,4),dtype=np.int64)
        for uid,tick in zip(senders,times):
            if tick < 6:dense[tick//2,list(ids).index(uid)]+=1
        a=grid.project([(ids,dense,None)],bin_seconds=.002)
        b=grid.project_events(senders,times,observed_unit_ids=ids,bin_edges_ticks=edges,dt_seconds=.001)
        for key in ("count_sum","observed_rate_hz","observed_unit_count","total_unit_count","complete"):
            np.testing.assert_equal(getattr(a.field,key),getattr(b.field,key))
        np.testing.assert_equal(a.density_hz_per_mm2,b.density_hz_per_mm2)
        self.assertEqual(b.field.count_sum.sum(),6)

    def test_random_events_equal_dense_chunk_readout(self):
        grid=self.grid();rng=np.random.default_rng(310);ids=grid.mapping.unit_ids
        chosen=rng.integers(0,4,400);times=rng.integers(0,40,400);edges=np.arange(0,41,4)
        dense=np.zeros((10,4),dtype=np.int64);np.add.at(dense,(times//4,chosen),1)
        a=grid.project([(ids[:2],dense[:,:2],None),(ids[2:],dense[:,2:],None)],bin_seconds=.004)
        b=grid.project_events(ids[chosen],times,observed_unit_ids=ids,bin_edges_ticks=edges,dt_seconds=.001)
        np.testing.assert_equal(a.field.count_sum,b.field.count_sum)
        np.testing.assert_equal(a.density_hz_per_mm2,b.density_hz_per_mm2)

    def test_unobserved_is_not_silent_and_known_empty_is_zero(self):
        grid=self.grid()
        b=grid.project_events(np.array([],dtype=int),np.array([],dtype=int),observed_unit_ids=[9],
                             bin_edges_ticks=[0,1],dt_seconds=.001)
        missing=(b.field.total_unit_count>0)&~b.field.complete[0]
        self.assertTrue(np.isnan(b.density_hz_per_mm2[0,missing]).all())
        self.assertTrue((b.density_hz_per_mm2[0,b.field.total_unit_count==0]==0).all())

    def test_rejects_unknown_or_unobserved_event_id(self):
        for uid in (123,2):
            with self.assertRaises(ValueError):self.grid().project_events([uid],[1],observed_unit_ids=[9],
                bin_edges_ticks=[0,2],dt_seconds=.001)

    def test_rejects_inconsistent_clocks_and_duplicate_observation_ids(self):
        for edges,observed in (([0,2,3],[9]),([0,0],[9]),([0.,1.],[9]),([0,1],[9,9])):
            with self.assertRaises(ValueError):self.grid().project_events([9],[0],observed_unit_ids=observed,
                bin_edges_ticks=edges,dt_seconds=.001)


if __name__ == "__main__":unittest.main()
