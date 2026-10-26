"""Recovery adapter checks committed observations without rerunning source cases."""
import json
from pathlib import Path
from types import SimpleNamespace, FunctionType, ModuleType
import tempfile
import unittest

import numpy as np

from .pfc_input_capture_recovery import restore_observations, observed_source_function
from .test_pfc_input_capture import NativeObservationBindingTests
from .pfc_factorized_source import SOURCE_ENTRY_SHA256
from .provenance import sha256_file


class ObservationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = NativeObservationBindingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.case = dict(self.fixture.case)
        self.case.update(initial_rng_seed=41, external_noise_seed=42,
                         source_rng_seed=42, random_stream_policy="independent_initial_external",
                         block_id="pfc_initial_41_external_42")
        self.config = dict(self.fixture.config, scope="source_reproduction",
                           random_stream_policy="independent_initial_external",
                           stimulus_strength_nA=.07,
                           cases=[dict(case_id="case", initial_rng_seed=41, external_noise_seed=42,
                                       stimulus_strength_nA=.07,distractor_strength_nA=0.)])
        self.native = dict(self.fixture.native, **{key:value for key,value in self.case.items()
                                                  if key not in self.fixture.native})
        self.native.update(source_rng_seed=42, initial_rng_seed=41, external_noise_seed=42)
        self.receipt = dict(self.fixture.receipt, case=self.case,
                            status="NATIVE_INPUT_OBSERVATION_CASE_COMPLETE", mode="audit",
                            case_id="case",source_entry_sha256=SOURCE_ENTRY_SHA256)
        directory=self.root/"inputs/case";directory.mkdir()
        for kind in ("terminal_state","numpy_rng","python_rng","generated_code"):
            path=directory/(kind+".npz")
            if kind=="terminal_state":np.savez(path,**self.fixture.arrays)
            else:path.write_bytes(b"fixture")
            self.receipt[kind+"_file"]="case/"+path.name
            self.receipt[kind+"_sha256"]=sha256_file(path)
        (directory/"observation.json").write_text(json.dumps(self.receipt))
        self.output=self.root/"recovered";self.output.mkdir()
        self.observer=SimpleNamespace(mode="audit",output=self.output/"inputs",_cases=[])
        self.observer.output.mkdir()

    def restore(self):
        return restore_observations(self.root,self.output,self.observer,self.config,
                                    {"case":dict(case=self.native)})

    def test_committed_case_restores_existing_receipt_and_snapshots(self):
        self.assertEqual(self.restore(),["case"])
        self.assertEqual(len(self.observer._cases),1)
        source=self.root/"inputs/case/observation.json"
        restored=self.observer.output/"case/observation.json"
        self.assertEqual(source.read_bytes(),restored.read_bytes())
        self.assertEqual(self.observer._cases[0]["receipt_sha256"],sha256_file(source))

    def test_changed_snapshot_and_wrong_mode_reject_recovery(self):
        path=self.root/"inputs/case/terminal_state.npz"
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"ARTIFACT_CHANGED"):
            self.restore()
        self.receipt["mode"]="capture"
        (self.root/"inputs/case/observation.json").write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError,"RECEIPT_CHANGED"):
            self.restore()

    def test_source_function_adapter_retains_globals_and_observes_only_missing_calls(self):
        module=ModuleType("fixture_source")
        exec("def simulate_wm(value):\n    return value + 1",vars(module))
        calls=[]
        observer=SimpleNamespace(begin_case=lambda case,config:calls.append(("begin",case)),
                                 finish_case=lambda case,result:calls.append(("finish",case,result)))
        original=module.simulate_wm
        wrapped=observed_source_function(module,observer,[dict(case_id="missing")],{})
        self.assertIsInstance(wrapped,FunctionType)
        self.assertIs(wrapped.__globals__,original.__globals__)
        self.assertEqual(wrapped(5),6)
        self.assertEqual([call[0] for call in calls],["begin","finish"])
        with self.assertRaisesRegex(ValueError,"UNDECLARED_SOURCE_CALL"):
            wrapped(5)


if __name__=="__main__":
    unittest.main()
