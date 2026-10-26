"""Native arithmetic witness tests, with exact known binary64 boundary answers."""
from fractions import Fraction as F
import hashlib
from pathlib import Path
import unittest

from population_dynamics import pfc_deterministic_replay as executor
from population_dynamics import pfc_numpy_arithmetic_witness as witness
from population_dynamics.pfc_synaptic_trace_bounds import NmdaParameters


class NumpyArithmeticWitnessTests(unittest.TestCase):
    def test_nearest_even_normal_boundaries(self):
        self.assertEqual(witness.nearest_even_bits(F(1) + F(1,2**53)), 0x3ff0000000000000)
        self.assertEqual(witness.nearest_even_bits(F(1) + F(3,2**53)), 0x3ff0000000000002)
        self.assertEqual(witness.nearest_even_bits(-F(1) - F(3,2**53)), 0xbff0000000000002)
        self.assertEqual(witness.nearest_even_bits(F(2) - F(1,2**53)), 0x4000000000000000)

    def test_gradual_underflow_and_signed_halfway_zero(self):
        self.assertEqual(witness.nearest_even_bits(F(1,2**1075)), 0)
        self.assertEqual(witness.nearest_even_bits(-F(1,2**1075)), 0x8000000000000000)
        self.assertEqual(witness.nearest_even_bits(F(3,2**1075)), 2)
        self.assertEqual(witness.nearest_even_bits(F(1,2**1022) - F(1,2**1075)), 0x0010000000000000)

    def test_known_binary64_decode_and_division(self):
        self.assertEqual(witness.fraction_from_bits(1), F(1,2**1074))
        self.assertEqual(witness.fraction_from_bits(0x3ff0000000000000), 1)
        self.assertEqual(witness.nearest_even_bits(F(1,3)), 0x3fd5555555555555)
        self.assertEqual(witness.nearest_even_bits(F(2,3)), 0x3fe5555555555555)

    def test_nonfinite_or_invalid_oracle_inputs_fail(self):
        with self.assertRaisesRegex(ValueError,"NONFINITE"):
            witness.nearest_even_bits(F(2**1024))
        with self.assertRaisesRegex(ValueError,"NONFINITE"):
            witness.fraction_from_bits(0x7ff0000000000000)
        with self.assertRaises(TypeError):
            witness.nearest_even_bits(0.5)

    def test_wrong_source_binding_fails_before_execution(self):
        p = NmdaParameters(F.from_float(1.25e-5),F.from_float(.00188),F.from_float(.065),F(500))
        with self.assertRaisesRegex(ValueError,"EXECUTOR_MISMATCH"):
            witness.source_witness(p,source_sha256="0"*64,refractory_ticks=160)

    def test_actual_source_width_and_unqualified_scope(self):
        constants=executor._constants({"weight_scaling_factor":1.0},"E")
        p=NmdaParameters(F.from_float(1.25e-5),F.from_float(constants["tau_nmda_x"]),
                         F.from_float(constants["tau_nmda_s"]),F.from_float(constants["alpha_nmda"]))
        source_sha=hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest()
        report=witness.source_witness(p,source_sha256=source_sha,refractory_ticks=160)
        self.assertEqual(report["status"],"PASS_BOUNDED_WITNESS")
        self.assertEqual(report["distinct_source_state_pairs"],49)
        self.assertEqual(report["source_cases"][-1]["size"],80000)
        self.assertFalse(report["native_backend_certified"])
        self.assertFalse(report["source_qualification_granted"])
        self.assertFalse(report["closed_network_fidelity_qualified"])
        self.assertEqual(report["production_episodes"],0)
        with self.assertRaisesRegex(ValueError,"REFRACTORY_MISMATCH"):
            witness.source_witness(p,source_sha256=source_sha,refractory_ticks=317)


if __name__ == "__main__":
    unittest.main()
