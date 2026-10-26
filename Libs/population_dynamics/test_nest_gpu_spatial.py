"""Source drive, state, repeated-edge and failure contracts for streamed replay."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from .nest_gpu_spatial import Execution, observe_continuous_state, prepare_graph, translated_batch
from .spatial_connectivity import EDGE_DTYPE, SCHEMA, SOURCE_REVISION


def fixture():
    names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I", "TC"]
    net = dict(neuron_model="iaf_psc_exp", dc_input=False, populations=names,
        num_neurons=[1]*8+[2], extent=1., DC_amp=list(range(8)), poisson_input=True,
        bg_rate=8., ext_indegrees=list(range(1,9)), thalamic_input=True,
        thalamic_input_type="pulses", th_radius=.5, th_pulse_start=3., th_interval=4.,
        th_delay_pulse_generator=1., neuron_params=dict(C_m=250., tau_m=10., E_L=-65.,
            t_ref=2., V_th=-50., V_reset=-65., tau_syn_ex=.5, tau_syn_in=.5))
    config = dict(simulation=dict(sim_resolution=.1, t_presim=1., t_sim=9.),
                  pulse_control=dict(decision_time_ms=3., future_emission_ms=[7.]))
    manifest = dict(schema=SCHEMA, status="GENERATED", source_revision=SOURCE_REVISION,
                    external_device_edges_included=False, resolved_source_network=net, source_config=config)
    layout = dict(unit_ids=np.arange(1,11), populations=np.asarray(names[:-1]+["TC", "TC"]),
                  positions_mm=np.asarray([[0.,0.]]*8+[[.3,.4],[.4,.4]]),
                  cortical_initial_vm_mv=np.arange(-52.,-44.))
    return manifest, layout


class SpatialExecutionTests(unittest.TestCase):
    def test_source_current_relative_voltage_and_untruncated_initial_state(self):
        m,a=fixture(); p=prepare_graph(m,a,Execution())
        np.testing.assert_array_equal(p["mapped"]["I_e"], np.arange(8))
        np.testing.assert_array_equal(p["mapped"]["V_m_rel"], a["cortical_initial_vm_mv"]+65)
        np.testing.assert_array_equal(p["rates"], 8*np.arange(1,9))
        self.assertGreater(p["mapped"]["V_m_rel"][-1],p["mapped"]["Theta_rel"][-1])

    def test_native_pulse_delay_mask_and_committed_emissions(self):
        m,a=fixture()
        effective=prepare_graph(m,a,Execution());initial=prepare_graph(m,a,Execution(schedule="initial"))
        np.testing.assert_array_equal(effective["tc_schedules"][9],[30,80])
        np.testing.assert_array_equal(initial["tc_schedules"][9],[30,70])
        self.assertEqual(len(effective["tc_schedules"][10]),0)

    def test_duration_extension_expands_default_schedule(self):
        m,a=fixture();m["source_config"].pop("pulse_control")
        p=prepare_graph(m,a,Execution(duration_ms=18.))
        np.testing.assert_array_equal(p["tc_schedules"][9],[30,70,110,150])
        self.assertEqual(p["duration"],18.)

    def test_rejects_changed_history_and_unqualified_control(self):
        for variant in ("past", "poisson", "dc", "refractory", "mask"):
            m,a=fixture();n=m["resolved_source_network"]
            if variant=="past":m["source_config"]["pulse_control"]["future_emission_ms"]=[2.]
            if variant=="poisson":n["thalamic_input_type"]="poisson"
            if variant=="dc":n["dc_input"]=True
            if variant=="refractory":n["neuron_params"]["t_ref"]=2.05
            if variant=="mask":n["th_radius"]=.6
            with self.subTest(variant=variant), self.assertRaises(ValueError):prepare_graph(m,a,Execution())

    def test_rejects_unknown_source_and_layout_mismatch(self):
        for variant in ("revision", "voltage", "identity", "position", "external"):
            m,a=fixture()
            if variant=="revision":m["source_revision"]="other"
            if variant=="voltage":a["cortical_initial_vm_mv"][0]=np.nan
            if variant=="identity":a["unit_ids"][0]=2
            if variant=="position":a["positions_mm"][0,0]=2.
            if variant=="external":m["external_device_edges_included"]=True
            with self.subTest(variant=variant),self.assertRaises(ValueError):prepare_graph(m,a,Execution())

    def test_preserves_repeated_edges_and_raw_native_delay_quantization(self):
        edges=np.array([(1,2,4.,.25,0,0),(1,2,5.,.35,1,1),(2,1,-3.,.2,0,0)],dtype=EDGE_DTYPE)
        rows=translated_batch(edges,.1)
        np.testing.assert_array_equal(rows["source"],[1,1,2])
        np.testing.assert_array_equal(rows["delay_ticks"],[3,4,2])
        np.testing.assert_array_equal(rows["weight"],[4.,5.,-3.])
        np.testing.assert_array_equal(rows["port"],[0,0,1])
        self.assertEqual(len(rows),3)

    def test_positive_execution_budgets(self):
        for settings in (Execution(edge_batch_size=0),Execution(readback_sources_per_population=0),
                         Execution(state_units_per_population=-1),Execution(duration_ms=float("nan"))):
            with self.assertRaises(ValueError):settings.validate()

    def test_real_cli_failed_input_has_nonzero_exit_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"config.yaml").write_text("edge_batch_size: 7\n")
            result=subprocess.run([sys.executable,"-m","Libs.population_dynamics.nest_gpu_spatial",
                "--graph",str(root/"missing"),"--config",str(root/"config.yaml"),"--output",str(root/"run")],
                text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            report=json.loads((root/"run/report.json").read_text())
            self.assertEqual(report["status"],"FAILED")
            self.assertEqual(report["error_type"],"FileNotFoundError")
            self.assertFalse(report["scientific_reference_qualified"])


class EpisodeOverrideTests(unittest.TestCase):
    def test_continuous_noise_reset_follows_last_committed_observation(self):
        clock={"tick":0};calls=[]
        def step():clock["tick"]+=1;return 0
        def read(backend,nodes,key):
            calls.append((clock["tick"],key))
            return np.full(len(nodes),.5 if key in ("P11ex","P11in") else clock["tick"])
        native=SimpleNamespace(GetErrorCode=lambda:0,GetErrorMessage=lambda:"",SetSimTime=lambda duration:None)
        library=SimpleNamespace(NESTGPU_StartSimulation=mock.Mock(return_value=0),
                                NESTGPU_SimulationStep=mock.Mock(side_effect=step))
        def reset():calls.append((clock["tick"],"reset"));return {"reset_tick":clock["tick"]}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("Libs.population_dynamics.nest_gpu_spatial.ctypes.CDLL",return_value=library), \
             mock.patch("Libs.population_dynamics.nest_gpu_spatial._read",side_effect=read), \
             mock.patch("Libs.population_dynamics.nest_gpu_spatial.sha256_file",return_value="fixture"), \
             mock.patch.dict("os.environ",{"NESTGPU_LIB":"fixture.so"}):
            result=observe_continuous_state(native,[1,2],np.array([1,2]),[2,4],.1,.6,Path(directory),
                noise_reset_tick=4,noise_reset_action=reset)
        self.assertEqual(clock["tick"],6)
        self.assertEqual(result["future_background_noise"],{"reset_tick":4})
        self.assertEqual([key for tick,key in calls if tick==4],
                         ["V_m_rel","I_syn_ex","I_syn_in","refractory_step","reset"])
        library.NESTGPU_StartSimulation.assert_called_once()

    def test_future_noise_requires_source_delay_and_remaining_future(self):
        m,a=fixture();m["resolved_source_network"]["delay_poisson"]=1.5
        settings=Execution(future_background_noise=dict(boundary_time_ms=3.,seeds_per_population=list(range(73012,73020))))
        prepare_graph(m,a,settings)
        for noise in (dict(boundary_time_ms=3.,seeds_per_population=[73012]*8),
                      dict(boundary_time_ms=9.,seeds_per_population=list(range(73012,73020))),
                      dict(boundary_time_ms=3.05,seeds_per_population=list(range(73012,73020)))):
            with self.subTest(noise=noise),self.assertRaises(ValueError):
                prepare_graph(m,a,Execution(future_background_noise=noise))
        m["resolved_source_network"]["poisson_input"]=False
        with self.assertRaisesRegex(ValueError,"source Poisson"):
            prepare_graph(m,a,settings)

    def test_pulse_program_overrides_only_episode_not_source_manifest(self):
        m,a=fixture();before=copy.deepcopy(m)
        p=prepare_graph(m,a,Execution(pulse_control={"initial_emission_ms":[4.]}))
        np.testing.assert_array_equal(p["tc_schedules"][9],[50])
        self.assertEqual(m,before)

    def test_empty_schedule_is_explicit_omission(self):
        m,a=fixture();p=prepare_graph(m,a,Execution(pulse_control={"initial_emission_ms":[]}))
        self.assertTrue(all(len(v)==0 for v in p["tc_schedules"].values()))

    def test_seed_and_initialization_provenance_required(self):
        for x in (Execution(background_seed=0),Execution(background_seed=True),
                  Execution(initial_state_seed=22),Execution(pulse_control=[])):
            with self.assertRaises(ValueError):x.validate()


if __name__ == "__main__":unittest.main()
