"""CPU allocation identities for Merlin and retained Slurm consumers."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from . import pfc_parameter_domain_summary as domain_summary
from . import pfc_parameter_task_dynamics as task_dynamics
from .provenance import cpu_execution_provenance


class CpuExecutionProvenanceTests(unittest.TestCase):
    def merlin(self):
        return {
            "RH2_JOB_RUN_ID": "run-123", "ARNOLD_TRIAL_ID": "trial-456",
            "ARNOLD_MONITOR_ROLE": "head", "ARNOLD_APPLIED_GPU_NUM": "0",
            "ARNOLD_HEAD_CPU": "16",
        }

    def slurm(self):
        return {"SLURM_JOB_ID": "123456", "SLURMD_NODENAME": "cpu05",
                "SLURM_CPUS_PER_TASK": "8"}

    def execute(self, environment):
        with patch.dict(os.environ, environment, clear=True), patch(
                "population_dynamics.provenance.socket.gethostname", return_value="actual-host"):
            return cpu_execution_provenance()

    def rejected(self, environment, field):
        with self.assertRaisesRegex(ValueError, "CPU_EXECUTION_INVALID_FIELD:" + field + "$"):
            self.execute(environment)

    def test_merlin_without_slurmd_records_real_identity(self):
        environment = self.merlin()
        environment["SLURM_JOB_ID"] = "merlin_trial-456_pfc_consumer_compatibility"
        self.assertEqual(self.execute(environment), {
            "backend": "merlin", "run_id": "run-123", "trial_id": "trial-456",
            "hostname": "actual-host", "allocated_cpus": 16, "allocated_gpus": 0,
        })

    def test_legacy_slurm_accepts_each_supported_cpu_node(self):
        for node in ("cpu03", "cpu04", "cpu05", "cpu06"):
            with self.subTest(node=node):
                environment = dict(self.slurm(), SLURMD_NODENAME=node)
                value = self.execute(environment)
                self.assertEqual(value, {
                    "backend": "slurm", "run_id": "123456", "trial_id": None,
                    "slurm_job_id": "123456", "slurm_node": node,
                    "hostname": "actual-host", "allocated_cpus": 8, "allocated_gpus": 0,
                })

    def test_legacy_slurm_cpu_count_on_node_and_unrelated_rh2_noise(self):
        environment = self.slurm()
        del environment["SLURM_CPUS_PER_TASK"]
        environment.update(SLURM_CPUS_ON_NODE="32", RH2_CHECKPOINT_HDFS="hdfs://checkpoint")
        self.assertEqual(self.execute(environment)["allocated_cpus"], 32)
        self.assertEqual(self.execute(environment)["backend"], "slurm")

    def test_compatibility_job_identity_is_rejected(self):
        self.rejected(dict(self.slurm(), SLURM_JOB_ID="merlin_trial_compatibility"), "SLURM_JOB_ID")

    def test_no_allocation_is_rejected(self):
        self.rejected({}, "SLURM_JOB_ID")

    def test_partial_merlin_identity_refuses_slurm_fallback(self):
        for missing in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID"):
            for empty in (False, True):
                with self.subTest(missing=missing, empty=empty):
                    environment = dict(self.slurm(), **self.merlin())
                    if empty:
                        environment[missing] = ""
                    else:
                        del environment[missing]
                    self.rejected(environment, missing)

    def test_merlin_worker_refuses_slurm_fallback(self):
        self.rejected(dict(self.slurm(), **dict(self.merlin(), ARNOLD_MONITOR_ROLE="worker")),
                      "ARNOLD_MONITOR_ROLE")

    def test_merlin_gpu_refuses_slurm_fallback(self):
        self.rejected(dict(self.slurm(), **dict(self.merlin(), ARNOLD_APPLIED_GPU_NUM="1")),
                      "ARNOLD_APPLIED_GPU_NUM")

    def test_visible_gpu_is_rejected_on_both_backends(self):
        for environment in (self.merlin(), self.slurm()):
            with self.subTest(backend="merlin" if "RH2_JOB_RUN_ID" in environment else "slurm"):
                self.rejected(dict(environment, CUDA_VISIBLE_DEVICES="0"), "CUDA_VISIBLE_DEVICES")

    def test_merlin_missing_gpu_declaration_is_rejected(self):
        environment = self.merlin()
        del environment["ARNOLD_APPLIED_GPU_NUM"]
        self.rejected(environment, "ARNOLD_APPLIED_GPU_NUM")

    def test_merlin_cpu_count_is_positive_integer_and_required(self):
        for value in (None, "", "0", "-1", "1.5", " 4", "four"):
            with self.subTest(value=value):
                environment = dict(self.slurm(), **self.merlin())
                if value is None:
                    del environment["ARNOLD_HEAD_CPU"]
                else:
                    environment["ARNOLD_HEAD_CPU"] = value
                self.rejected(environment, "ARNOLD_HEAD_CPU")

    def test_legacy_slurm_requires_node_and_cpu_count(self):
        self.rejected(dict(self.slurm(), SLURMD_NODENAME="mgmt2"), "SLURMD_NODENAME")
        self.rejected(dict(self.slurm(), SLURM_CPUS_PER_TASK="0"), "SLURM_CPUS_PER_TASK")
        environment = self.slurm()
        del environment["SLURM_CPUS_PER_TASK"]
        self.rejected(environment, "SLURM_CPUS_ON_NODE")

    def test_complete_cohort_gate_remains_required_on_each_backend(self):
        plan = dict(schema="pfc_parameter_domain_quality_v1", scope="development",
                    block_indices=[6, 7, 8], planned_complete_episodes=3000,
                    complete_episodes=0, production_episodes=0,
                    source_quality_qualified=False, task_sampling_qualified=False,
                    scientific_claim_qualified=False, production_eligible=False,
                    circuit_split_counts=dict(train=70, validation=10, calibration=10, test=10),
                    rows=[])
        for environment in (self.merlin(), self.slurm()):
            with self.subTest(backend="merlin" if "RH2_JOB_RUN_ID" in environment else "slurm"):
                self.execute(environment)
                with patch.dict(os.environ, environment, clear=True), patch.object(
                        domain_summary.quality, "read_json", return_value=plan):
                    with self.assertRaisesRegex(ValueError, "PFC_DOMAIN_COMPLETE3000_RECIPES_REQUIRED"):
                        domain_summary.aggregate(Path("prepared"), Path("observations.json"), Path("output"))
                with self.assertRaisesRegex(ValueError, "PFC_TASK_COMPLETE3000_DESCRIPTIVE_SUMMARY_REQUIRED"):
                    task_dynamics.validate_summary({"complete_development_episodes": 2999})


if __name__ == "__main__":
    unittest.main()
