"""Factorial timing and whole-block statistical validation contracts."""
import copy
import unittest
import numpy as np

from .pulse_history import conditions
from .pulse_history_analysis import block_prediction, block_summary, conditional_ridge


def design():
    config=dict(schema="mesocircuit_pulse_history_development_v1",initial_state_seeds=[101,102],
        background_seeds=[201,202],focal_arrivals_ms=[1000.,1500.],conditioning_offsets_ms=[-120.,-80.,-40.],
        history_window_ms=200.,response_window_ms=20.,response_curve_ms=80.,duration_ms=1600.,
        expected_episodes=16,run_order_seed=17,execution={})
    graph=dict(resolved_source_network=dict(th_delay_pulse_generator=1.),
               source_config=dict(simulation=dict(sim_resolution=.1)))
    return config,graph


class FactorialTests(unittest.TestCase):
    def test_full_crossed_denominator_and_exact_omission_pairing(self):
        c,g=design();rows=conditions(c,g,"rng.so")
        self.assertEqual(len(rows),16)
        lookup={(r["block"],r["focal_arrival_ms"],r["history"],r["present"]):r for r in rows}
        self.assertEqual(len(lookup),16)
        for block in range(2):
            for time in c["focal_arrivals_ms"]:
                for history in (False,True):
                    a=lookup[(block,time,history,False)]["settings"]
                    b=lookup[(block,time,history,True)]["settings"]
                    self.assertEqual(a["pulse_control"]["initial_emission_ms"],b["pulse_control"]["initial_emission_ms"][:-1])
                    self.assertEqual(b["pulse_control"]["initial_emission_ms"][-1],time-1.)
                    self.assertEqual(a["initial_state_seed"],b["initial_state_seed"])
                    self.assertEqual(a["background_seed"],b["background_seed"])
        self.assertEqual(rows,conditions(c,g,"rng.so"))

    def test_rejects_lost_conditions_or_truncated_response(self):
        c,g=design()
        for field,value in [("expected_episodes",15),("duration_ms",1550.),("background_seeds",[201,201])]:
            x=copy.deepcopy(c);x[field]=value
            with self.assertRaises(ValueError):conditions(x,g,"rng.so")


class ProbeTests(unittest.TestCase):
    def test_unpenalized_cell_means_are_shared_by_both_models(self):
        cell=np.tile(np.arange(4),6);x=np.zeros((24,3));y=cell.astype(float)*10
        np.testing.assert_allclose(conditional_ridge(x,y,cell,x,cell,100.),y,atol=1e-12)
        np.testing.assert_allclose(conditional_ridge(x[:,:0],y,cell,x[:,:0],cell,1.),y,atol=1e-12)

    def test_outer_held_out_targets_never_affect_own_fit_or_selection(self):
        rng=np.random.default_rng(31);blocks=np.repeat(np.arange(6),4);cells=np.tile(np.arange(4),6)
        x=rng.normal(size=(24,3));y=cells*3+x[:,0]*2
        pred,records=block_prediction(x,y,blocks,cells)
        changed=y.copy();changed[blocks==0]+=10000
        other,record2=block_prediction(x,changed,blocks,cells)
        np.testing.assert_array_equal(pred[blocks==0],other[blocks==0])
        self.assertEqual(records[0]["inner_validation_mse"],record2[0]["inner_validation_mse"])
        self.assertEqual(records[0]["selected_ridge"],record2[0]["selected_ridge"])
        self.assertLess(np.mean((pred[:,1]-y)**2),np.mean((pred[:,0]-y)**2))

    def test_exhaustive_bootstrap_preserves_independent_block_denominator(self):
        r=block_summary([2.,2.,2.,2.,2.,2.])
        self.assertEqual(r["independent_blocks"],6)
        self.assertEqual(r["empirical_block_bootstrap_percentile_95"],[2.,2.])
        self.assertEqual(r["sign_symmetry_reference_p"],2/64)
        self.assertEqual(block_summary([0.]*6)["sign_symmetry_reference_p"],1.)


if __name__=="__main__":unittest.main()
