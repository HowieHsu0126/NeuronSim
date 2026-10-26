"""Keep failed-source transport distinct from completed scientific execution."""
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "return_appworld_merlin_result.py"
SPEC = importlib.util.spec_from_file_location("appworld_result_return", SCRIPT)
RETURN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETURN)
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "appworld_merlin_bootstrap", SCRIPT.with_name("run_appworld_benchmark_merlin.py"))
BOOTSTRAP = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(BOOTSTRAP)


class SourceOutcomeTests(unittest.TestCase):
    def inputs(self, code=0, status="COMPLETE", **extra):
        return {"source_receipt": {"exit_code": code, "status": status,
                                   "persisted": True}, **extra}

    def test_completed_source_remains_default(self):
        self.assertEqual(RETURN.validate_source(self.inputs()), "completed")

    def test_failed_source_requires_explicit_mode(self):
        with self.assertRaisesRegex(ValueError, "Completed persisted source required"):
            RETURN.validate_source(self.inputs(code=1, status="FAILED"))

    def test_explicit_failed_diagnostics_retains_source_exit(self):
        inputs = self.inputs(code=1, status="FAILED", source_outcome="failed_diagnostics")
        self.assertEqual(RETURN.validate_source(inputs), "failed_diagnostics")
        self.assertEqual(inputs["source_receipt"]["exit_code"], 1)
        self.assertEqual(inputs["source_receipt"]["status"], "FAILED")

    def test_success_cannot_be_classified_as_failure(self):
        with self.assertRaisesRegex(ValueError, "Failed persisted source required"):
            RETURN.validate_source(self.inputs(source_outcome="failed_diagnostics"))

    def test_nonterminal_failure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Failed persisted source required"):
            RETURN.validate_source(self.inputs(code=1, status="RUNNING",
                                              source_outcome="failed_diagnostics"))

    def test_unpersisted_source_is_rejected(self):
        inputs = self.inputs(code=1, status="FAILED", source_outcome="failed_diagnostics")
        inputs["source_receipt"]["persisted"] = False
        with self.assertRaisesRegex(ValueError, "Persisted source required"):
            RETURN.validate_source(inputs)

    def test_unobserved_or_coerced_exit_code_is_rejected(self):
        for code in (None, True, "1"):
            with self.subTest(code=code), self.assertRaisesRegex(
                    ValueError, "Integer source exit code required"):
                RETURN.validate_source(self.inputs(code=code, status="FAILED",
                                                  source_outcome="failed_diagnostics"))

    def test_unknown_outcome_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported source outcome"):
            RETURN.validate_source(self.inputs(source_outcome="unknown"))


class CugaSourceClosureTests(unittest.TestCase):
    def inputs(self):
        return {
            "cuga_runtime": {"schema": "appworld_cuga_merlin_preparation_v1",
                             "persisted": True, "runtime_prepared": True,
                             "runtime_persisted": True},
            "cuga_prior_launcher": {"bytes": BOOTSTRAP.CUGA_PRIOR_LAUNCHER_BYTES,
                                    "sha256": BOOTSTRAP.CUGA_PRIOR_LAUNCHER_SHA256},
            "files": {name: {} for name in BOOTSTRAP.CUGA_SOURCE_FILES},
        }

    def test_complete_source_closure_is_accepted(self):
        BOOTSTRAP.validate_cuga_inputs(self.inputs())

    def test_missing_native_binding_is_rejected_before_restore(self):
        inputs = self.inputs()
        del inputs["files"]["Libs/population_dynamics/appworld_benchmark_native_binding.py"]
        with self.assertRaisesRegex(ValueError, "CUGA_DEVELOPMENT_SOURCE_SET_REQUIRED"):
            BOOTSTRAP.validate_cuga_inputs(inputs)


if __name__ == "__main__":
    unittest.main()
