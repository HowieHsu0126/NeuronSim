"""Causal source schedules for the continuous native control interface."""
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
from . import nest_gpu_live_control_check as canary
from .mesocircuit_continuation import PulseProtocol
from .nest_gpu_live_control import (future_poisson_reset_tick, remaining_tc_arrivals,
                                   reset_uniform_poisson_future)
from .nest_gpu_spatial import Execution, prepare_graph
from .test_nest_gpu_spatial import fixture


class LiveControlTests(unittest.TestCase):
    def test_future_poisson_switch_retains_committed_arrivals(self):
        self.assertEqual(future_poisson_reset_tick(10000, 17), 10015)
        self.assertEqual(future_poisson_reset_tick(100, 2), 100)
        for boundary, delay in ((0, 17), (100, 1), (True, 17), (100, 17.0)):
            with self.subTest(boundary=boundary, delay=delay), self.assertRaises(ValueError):
                future_poisson_reset_tick(boundary, delay)

    def test_future_poisson_switch_checks_actual_all_outgoing_edges(self):
        native = SimpleNamespace(GetConnections=mock.Mock(return_value="all-source-edges"),
            GetConnectionStatus=mock.Mock(return_value=[
                dict(source=9, target=1, delay=1.7), dict(source=9, target=2, delay=1.7)]),
            GetErrorCode=lambda: 0, GetErrorMessage=lambda: "")
        function = mock.Mock(return_value=0)
        library = SimpleNamespace(NeuronSim_ResetFuturePoissonRng=function)
        with mock.patch("Libs.population_dynamics.nest_gpu_live_control.ctypes.CDLL", return_value=library), \
             mock.patch.dict(canary.os.environ, {"NESTGPU_LIB": "fixture.so"}):
            result = reset_uniform_poisson_future(native, 9, [1, 2], 100, 17, 115, 73012, .1)
        function.assert_called_once_with(9, 73012, 17, 115)
        native.GetConnections.assert_called_once_with(source=[9])
        self.assertEqual(result["first_resampled_arrival_tick"], 116)
        self.assertFalse(result["state_and_queues_reinitialized"])

    def test_invalid_future_poisson_switch_fails_before_rng_mutation(self):
        function = mock.Mock(return_value=0)
        native = SimpleNamespace(GetConnections=mock.Mock(return_value="all-source-edges"),
            GetConnectionStatus=mock.Mock(), GetErrorCode=lambda: 0, GetErrorMessage=lambda: "")
        bad_edges = ([dict(source=9, target=1, delay=1.6), dict(source=9, target=2, delay=1.7)],
                     [dict(source=9, target=1, delay=1.7)],
                     [dict(source=9, target=1, delay=1.7), dict(source=9, target=3, delay=1.7)])
        with mock.patch("Libs.population_dynamics.nest_gpu_live_control.ctypes.CDLL",
                        return_value=SimpleNamespace(NeuronSim_ResetFuturePoissonRng=function)), \
             mock.patch.dict(canary.os.environ, {"NESTGPU_LIB": "fixture.so"}):
            for edges in bad_edges:
                native.GetConnectionStatus.return_value = edges
                with self.subTest(edges=edges), self.assertRaisesRegex(ValueError, "uniform-delay"):
                    reset_uniform_poisson_future(native, 9, [1, 2], 100, 17, 115, 73012, .1)
            with self.assertRaises(ValueError):
                reset_uniform_poisson_future(native, 9, [1, 2], 100, 17, 114, 73012, .1)
        function.assert_not_called()

    def test_clear_retains_boundary_emission_and_inflight_arrival(self):
        p = PulseProtocol(.1, 10, (20, 90, 95, 100, 101, 110, 140), True, 100, ())
        np.testing.assert_array_equal(remaining_tc_arrivals(p), [105, 110])

    def test_short_replacement_keeps_committed_pending_prefix(self):
        p = PulseProtocol(.1, 10, (20, 90, 95, 100, 101, 110, 140), True, 100, (102, 120, 140))
        np.testing.assert_array_equal(remaining_tc_arrivals(p), [105, 110, 112, 130, 150])

    def test_initial_empty_accepts_future_through_explicit_mask(self):
        p = PulseProtocol(.1, 10, (), True, 100, (101,))
        np.testing.assert_array_equal(remaining_tc_arrivals(p), [111])

    def test_live_graph_preloads_original_and_retains_effective_protocol(self):
        m, a = fixture()
        p = prepare_graph(m, a, Execution(live_pulse_control=True))
        np.testing.assert_array_equal(p['tc_schedules'][9], [30, 70])
        np.testing.assert_array_equal(p['stimulated_tc_mask'], [True, False])
        self.assertEqual(p['pulse_protocol'].effective_ticks, (20, 70))

    def test_explicit_mask_survives_empty_original_schedule(self):
        m, a = fixture()
        settings = Execution(live_pulse_control=True, pulse_control=dict(
            initial_emission_ms=[], decision_time_ms=3., future_emission_ms=[4.]))
        p = prepare_graph(m, a, settings)
        self.assertEqual(p['tc_schedules'][9].size, 0)
        np.testing.assert_array_equal(p['stimulated_tc_mask'], [True, False])

    def test_rejects_live_without_decision_or_initial_only_mode(self):
        m, a = fixture()
        for x in [Execution(live_pulse_control=True, schedule='initial'),
                  Execution(live_pulse_control=True, pulse_control=dict(initial_emission_ms=[])),
                  Execution(live_pulse_control='yes')]:
            with self.subTest(settings=x), self.assertRaises(ValueError):
                prepare_graph(m, a, x)


class NativeGroupTypesTests(unittest.TestCase):
    def test_native_canary_retains_NodeSeq_for_recording_and_readback(self):
        # The pinned nestgpu wrapper requires the exact NodeSeq type for these
        # group APIs. Lists remain valid for individual status and connections.
        class NodeSeq:
            def __init__(self, first, count): self.i0,self.n=first,count
            def __getitem__(self, index): return self.i0+index
            def __len__(self): return self.n
        groups=[];activated=[];recorded=[];read=[]
        def create(model,count):
            first=sum(len(group) for group in groups)
            group=NodeSeq(first,count);groups.append(group);return group
        def require_group(group):
            self.assertIs(type(group),NodeSeq)
        def activate(group): require_group(group);activated.append(group)
        def record(group,limit): require_group(group);self.assertEqual(limit,10000);recorded.append(group)
        def events(group): require_group(group);read.append(group);return [[] for _ in range(len(group))]
        def status(nodes,field): return np.zeros((len(nodes),1))
        backend=SimpleNamespace(Create=create,SetTimeResolution=lambda *a:None,
            SetRandomSeed=lambda *a:None,SetStatus=lambda *a:None,Connect=lambda *a:None,
            ActivateSpikeCount=activate,ActivateRecSpikeTimes=record,
            CreateRecord=lambda *a:0,SetSimTime=lambda *a:None,Calibrate=lambda:None,
            GetErrorCode=lambda:0,GetErrorMessage=lambda:'',GetNeuronStatus=status,
            GetRecSpikeTimes=events,GetRecordData=lambda *a:np.zeros((201,33)))
        library=SimpleNamespace(NESTGPU_StartSimulation=mock.Mock(return_value=0),
                                NESTGPU_SimulationStep=mock.Mock(return_value=0))
        with mock.patch.object(canary.importlib,'import_module',return_value=backend), \
             mock.patch.object(canary.ctypes,'CDLL',return_value=library), \
             mock.patch.object(canary,'_sha256',return_value='fixture'), \
             mock.patch.dict(canary.os.environ,{'NESTGPU_LIB':'fixture.so'}):
            arrays,report=canary._run_native('unchanged','preloaded')
        self.assertEqual(activated,groups);self.assertEqual(recorded,groups)
        self.assertEqual(read,[groups[1],groups[0],groups[2]])
        self.assertEqual([len(group) for group in groups],[8,3,1])
        self.assertEqual(arrays['cortex_spike_count'].shape,(8,))
        self.assertEqual(report['native_start_calls'],1)
        library.NESTGPU_StartSimulation.assert_called_once()
        self.assertEqual(library.NESTGPU_SimulationStep.call_count,200)


if __name__ == '__main__':
    unittest.main()
