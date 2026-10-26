"""Domain and information-boundary checks for native snapshot replay."""
import copy
import unittest

import numpy as np

from .nest_gpu_network import prepare, ticks, SOURCE_REVISION
from .network_snapshot import NEURON_PARAMETERS
from .nest_reference import EDGE_DTYPE


def fixture():
    metadata = {"source_revision":SOURCE_REVISION,"cortical_neuron_model":"iaf_psc_exp",
                "thalamic_neuron_model":"parrot_neuron","dt_ms":.1,"duration_ms":10.,
                "config":{"pulse_control":{"decision_time_ms":3.}},
                "devices":[{"id":10,"model":"poisson_generator","parameters":
                            {"rate":128.,"origin":0.,"start":0.,"stop":1e30}},
                           {"id":11,"model":"spike_generator","parameters":
                            {"initial_spike_times_ms":[2.,4.],"effective_spike_times_ms":[2.,5.]}}]}
    params = dict(C_m=250.,tau_m=10.,tau_syn_ex=.5,tau_syn_in=.5,E_L=-65.,V_th=-50.,
                  V_reset=-65.,t_ref=2.,I_e=50.,V_m=-63.)
    arrays = {"unit_ids":np.array([1,2,3]),"populations":np.array(["L23E","L23I","TC"]),
              "positions_mm":np.array([[.1,.2],[.2,.3],[.3,.4]]),
              "cortical_parameters":np.array([[params[k] for k in NEURON_PARAMETERS]]*2),
              "edges":np.array([(1,2,-4.,.3),(1,2,-5.,.4),(2,3,2.,.5),
                                (1,10,1.,1.5),(3,11,1.,1.)],dtype=EDGE_DTYPE)}
    return metadata, arrays


class NetworkPreparationTests(unittest.TestCase):
    def test_preserves_signed_multiedges_and_folds_source_delay(self):
        m,a=fixture(); p=prepare(m,a,"effective")
        np.testing.assert_array_equal(p["tc_schedules"][3],[30,60])
        np.testing.assert_array_equal(p["edges"],a["edges"][:4])
        self.assertEqual(p["folded_relay_edges"],1)
        np.testing.assert_array_equal(p["mapped_parameters"]["V_m_rel"],[2.,2.])
        np.testing.assert_array_equal(prepare(m,a,"initial")["tc_schedules"][3],[30,50])

    def test_rejects_changed_past_and_unmarked_control(self):
        for remove in [False,True]:
            m,a=fixture()
            if remove:m["config"].pop("pulse_control")
            else:m["devices"][1]["parameters"]["effective_spike_times_ms"][0]=2.1
            with self.assertRaises(ValueError):prepare(m,a,"effective")

    def test_rejects_recurrent_input_to_folded_relay(self):
        m,a=fixture();a["edges"][-1]["source"]=1
        with self.assertRaisesRegex(ValueError,"deterministic input"):prepare(m,a,"effective")

    def test_rejects_off_grid_refractory_and_delay(self):
        for field in ["refractory","delay"]:
            m,a=fixture()
            if field=="refractory":a["cortical_parameters"][0,NEURON_PARAMETERS.index("t_ref")]=.15
            else:a["edges"][0]["delay"]=.35
            with self.assertRaisesRegex(ValueError,"native tick"):prepare(m,a,"effective")

    def test_rejects_unqualified_finite_poisson_or_other_device(self):
        for variant in ["finite","dc"]:
            m,a=fixture()
            if variant=="finite":m["devices"][0]["parameters"]["stop"]=5.
            else:m["devices"][0]["model"]="dc_generator"
            with self.assertRaises(ValueError):prepare(m,a,"effective")

    def test_supports_explicit_infinity_from_snapshot_schema(self):
        m,a=fixture();m["devices"][0]["parameters"]["stop"]="positive_infinity"
        self.assertEqual(len(prepare(m,a,"effective")["background"]),1)

    def test_rejects_unknown_revision_and_identity(self):
        for variant in ["revision","identity"]:
            m,a=fixture()
            if variant=="revision":m["source_revision"]="unreviewed"
            else:a["edges"][0]["source"]=999
            with self.assertRaises(ValueError):prepare(m,a,"effective")

    def test_recorded_ticks_accept_rounding_without_shifting_spikes(self):
        times=np.array([.1,61.2,150.]).astype(np.float32).astype(float)
        np.testing.assert_array_equal(ticks(times,.1,recorded=True),[1,612,1500])
        with self.assertRaises(ValueError):ticks([61.25],.1,recorded=True)


if __name__ == "__main__":unittest.main()
