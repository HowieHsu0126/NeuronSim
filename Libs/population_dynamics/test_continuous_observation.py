import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .nest_gpu_spatial import Execution, observe_continuous_state
from .pulse_history import conditions
from .test_pulse_history import design


class Call:
    def __init__(self,callback):self.callback=callback
    def __call__(self):return self.callback()


class ObservationTests(unittest.TestCase):
    def test_single_start_preserves_step_sequence_and_requested_ticks(self):
        class Backend:
            def __init__(self):self.tick=0;self.starts=0
            def SetSimTime(self,value):self.duration=value
            def GetErrorCode(self):return 0
            def GetNeuronStatus(self,nodes,key):
                return [self.tick if key=='V_m_rel' else (1. if key.startswith('P11') else 0.) for _ in nodes]
            def start(self):self.starts+=1;return 0
            def step(self):self.tick+=1;return 0
        backend=Backend()
        class Library:pass
        library=Library();library.NESTGPU_StartSimulation=Call(backend.start);library.NESTGPU_SimulationStep=Call(backend.step)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lib=root/'native.so';lib.write_bytes(b'fixture')
            with patch.dict(os.environ,{'NESTGPU_LIB':str(lib)}),patch('ctypes.CDLL',return_value=library):
                receipt=observe_continuous_state(backend,[0,1],np.array([7,9]),[2,7],.1,1.,root)
            self.assertEqual(backend.starts,1);self.assertEqual(backend.tick,10)
            self.assertEqual([r['tick'] for r in receipt['observations']],[2,7])
            for record in receipt['observations']:
                with np.load(root/record['path']) as a:np.testing.assert_array_equal(a['V_m_rel'],[record['tick']]*2)

    def test_focal_observation_is_strictly_before_emission_and_optional(self):
        config,graph=design();plain=conditions(config,graph,'rng.so')
        self.assertTrue(all('state_observation_ticks' not in case['settings'] for case in plain))
        config['capture_focal_state']=True
        for case in conditions(config,graph,'rng.so'):
            expected=round((case['focal_arrival_ms']-1.)/.1)-1
            self.assertEqual(case['settings']['state_observation_ticks'],[expected])

    def test_observation_ticks_reject_repeats_and_boolean_times(self):
        for values in [[2,2],[3,1],[True],[-1]]:
            with self.assertRaises(ValueError):Execution(state_observation_ticks=values).validate()


if __name__=='__main__':unittest.main()
