"""CA3 candidate wrapper checks; run against the staged frozen dependency closure."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ca3_merlin_remote as remote
import prepare_ca3_merlin_replay as prepare


@unittest.skipUnless(os.environ.get("CA3_PREFLIGHT_JOB"), "Requires the staged frozen CA3 preflight inputs")
class TestCA3MerlinReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = json.loads(Path(os.environ["CA3_PREFLIGHT_JOB"]).read_text())
        cls.job.pop("execution_mode", None)
        cls.job.pop("persistence_batch", None)
        cls.job.pop("completion75_recovery", None)
        cls.job.pop("condition_index", None)

    def test_original_command_preserves_original_runtime_and_reference(self):
        job = deepcopy(self.job)
        job.pop("candidate_runtime")
        command = remote.source_command(job, Path("/tmp/episode"))
        self.assertNotIn("--runtime-composition", command)
        self.assertEqual(command[command.index("--packaged") + 1],
                         job["execution_inputs"]["prepared"] + "/ca3_runtime")
        self.assertEqual(command[command.index("--prefix-reference") + 1],
                         job["execution_inputs"]["reference"])

    def test_candidate_command_requires_composed_execution(self):
        job = deepcopy(self.job)
        with self.assertRaises(KeyError):
            remote.source_command(job, Path("/tmp/episode"))
        job["candidate_execution"] = dict(packaged="/tmp/candidate_runtime",
            source_repo="/tmp/candidate_source", runtime_composition="/tmp/composition.json")
        command = remote.source_command(job, Path("/tmp/episode"))
        self.assertEqual(command[command.index("--packaged") + 1], "/tmp/candidate_runtime")
        self.assertEqual(command[-2:], ["--runtime-composition", "/tmp/composition.json"])
        self.assertEqual(command[command.index("--condition-index") + 1], "3")

    def test_candidate_upload_cannot_overwrite_frozen_inputs(self):
        candidate = deepcopy(self.job["candidate_runtime"])
        prepare.validate_candidate(candidate, self.job["input_files"])
        key, binding = next(iter(self.job["input_files"].items()))
        candidate["input_files"][key] = binding
        with self.assertRaisesRegex(ValueError, "MUST_NOT_REPLACE_FROZEN_INPUTS"):
            prepare.validate_candidate(candidate, self.job["input_files"])

    def test_candidate_requires_all_bound_artifacts(self):
        for role in ("library", "build_report", "receipt", "consumer"):
            with self.subTest(role=role):
                candidate = deepcopy(self.job["candidate_runtime"])
                candidate[role]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "CANDIDATE_INPUT_BINDING_DIFFERS"):
                    prepare.validate_candidate(candidate, self.job["input_files"])

    def test_preflight_binds_image_code_and_inputs(self):
        candidate = self.job["candidate_runtime"]
        identity, image = self.job["execution_inputs"], self.job["image_meta"]
        code = Path(remote.__file__).read_text()
        proof = dict(status="PASS_CA3_FROZEN_CANDIDATE_PREFLIGHT", exit_code=0, persisted=True,
            candidate_input_files=candidate["input_files"], source_file_bindings=identity["source_file_bindings"],
            runtime_build_sha256=identity["runtime_build_sha256"], image_meta=image,
            remote_sha256=hashlib.sha256(code.encode()).hexdigest())
        prepare.validate_candidate_preflight(candidate, proof, identity, image, code)
        for key, value in (("exit_code", 1), ("persisted", False), ("image_meta", {}),
                           ("remote_sha256", "0" * 64), ("candidate_input_files", {}),
                           ("source_file_bindings", {})):
            with self.subTest(field=key):
                changed = deepcopy(proof)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "MATCHING_CPU_CANDIDATE_PREFLIGHT_REQUIRED"):
                    prepare.validate_candidate_preflight(candidate, changed, identity, image, code)

    def test_consumer_overlay_preserves_other_functions_and_dataset_identity_call(self):
        old = Path(os.environ["CA3_FROZEN_CONSUMER"]).read_text()
        current = Path(os.environ["CA3_CURRENT_CONSUMER"]).read_text()
        changed = prepare.candidate_consumer(old, current)
        self.assertEqual(changed, Path(self.job["candidate_runtime"]["consumer"]["path"]).read_text())
        old_nodes = {node.name: node for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
        new_nodes = {node.name: node for node in ast.parse(changed).body if isinstance(node, ast.FunctionDef)}
        for name in old_nodes.keys() - {"run", "main"}:
            self.assertEqual(ast.dump(old_nodes[name]), ast.dump(new_nodes[name]), name)
        def identity_calls(node):
            return [ast.dump(call) for call in ast.walk(node) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name) and call.func.id == "episode_identities"]
        self.assertEqual(identity_calls(old_nodes["run"]), identity_calls(new_nodes["run"]))
        self.assertNotIn("production_identity", changed)
        with self.assertRaisesRegex(ValueError, "PATCH_CONTEXT_DIFFERS"):
            prepare.candidate_consumer(old.replace("def run(packaged:", "def different(packaged:"), current)

    def test_all_exact_prefix_and_replay_gates_remain_required(self):
        job = deepcopy(self.job)
        job.pop("candidate_runtime")
        from population_dynamics import ca3_benchmark_readout
        prefix = dict(prefix_intervals_compared=1001, extended_intervals_verified=1210,
            unit_millisecond_counts_compared=168106 * 1001, current_values_compared=168106 * 1001,
            exact_float32_bits_equal=True, counts_exact=True, original_terminal_footer_verified=True,
            extended_terminal_footer_verified=True, reference=job["execution_inputs"]["reference"])
        bound = dict(context={}, native=dict(duration_ms=1210, production_episodes=0,
            independent_current_replay=dict(exact_float32_bits_equal=True), original_protocol_prefix=prefix,
            condition={}, units=168106, events=0))
        with patch.object(ca3_benchmark_readout, "bind_episode", return_value=bound):
            result = remote.bind_result(job, Path("/tmp/episode"))
        self.assertEqual(result["production_episodes"], 0)
        self.assertFalse(result["scientific_fidelity_qualified"])
        for key in prefix:
            with self.subTest(gate=key):
                changed = deepcopy(bound)
                changed["native"]["original_protocol_prefix"][key] = None
                with patch.object(ca3_benchmark_readout, "bind_episode", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "FROZEN_PREFIX_OR_CURRENT_REPLAY_GATE_FAILED"):
                        remote.bind_result(job, Path("/tmp/episode"))
        changed = deepcopy(bound)
        changed["native"]["independent_current_replay"]["exact_float32_bits_equal"] = False
        with patch.object(ca3_benchmark_readout, "bind_episode", return_value=changed):
            with self.assertRaisesRegex(ValueError, "FROZEN_PREFIX_OR_CURRENT_REPLAY_GATE_FAILED"):
                remote.bind_result(job, Path("/tmp/episode"))



@unittest.skipUnless(os.environ.get("CA3_PERSISTENCE_JOB"), "Requires the bound persistence-condition inputs")
class TestCA3PersistenceConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = json.loads(Path(os.environ["CA3_PERSISTENCE_JOB"]).read_text())
        cls.job["candidate_execution"] = dict(source_repo=os.environ["CA3_COMPOSED_SOURCE"],
            packaged="/tmp/fixture_runtime", runtime_composition="/tmp/fixture_composition.json")

    def native_fixture(self):
        native = remote.bound_document(self.job["persistence_batch"]["prerequisite"]["episode_report"])
        index = self.job["persistence_batch"]["conditions"][0]["condition_index"]
        identity, condition = remote.persistence_condition_inputs(self.job, index)
        native["candidate_identity"], native["condition"] = identity, condition
        return native, identity, condition

    def test_bound_prerequisite_and_commands(self):
        proof = remote.validate_persistence_batch(self.job)
        self.assertEqual(proof["condition_indices"],
                         [row["condition_index"] for row in self.job["persistence_batch"]["conditions"]])
        for index in proof["condition_indices"]:
            command = remote.source_command(dict(self.job, condition_index=index), Path("/tmp/episode"))
            self.assertEqual(command[command.index("--condition-index") + 1], str(index))
            self.assertNotIn("--prefix-reference", command)
            self.assertIn("--current-replay-engine", command)

    def test_completed_or_excluded_conditions_rejected(self):
        for index in (3, 4, True, 10):
            job = deepcopy(self.job)
            job["persistence_batch"]["conditions"][0]["condition_index"] = index
            with self.assertRaisesRegex(ValueError, "CONDITION_SET_DIFFERS"):
                remote.validate_persistence_batch(job)

    def test_duplicate_condition_rejected(self):
        job = deepcopy(self.job)
        job["persistence_batch"]["conditions"].append(deepcopy(job["persistence_batch"]["conditions"][0]))
        with self.assertRaisesRegex(ValueError, "CONDITION_SET_DIFFERS"):
            remote.validate_persistence_batch(job)

    def test_real_merlin_origin_is_bound(self):
        for field in ("merlin_job_run_id", "merlin_trial_id", "result_sha256"):
            job = deepcopy(self.job)
            job["persistence_batch"]["prerequisite"]["origin"][field] = "different"
            with self.assertRaisesRegex(ValueError, "REAL_MERLIN_PREFIX_PREREQUISITE"):
                remote.validate_persistence_batch(job)

    def test_document_bytes_are_bound(self):
        job = deepcopy(self.job)
        job["persistence_batch"]["prerequisite"]["episode_report"]["content"] += " "
        with self.assertRaisesRegex(ValueError, "DOCUMENT_BINDING_DIFFERS"):
            remote.validate_persistence_batch(job)

    def test_all_full_episode_gates_are_required(self):
        native, identity, condition = self.native_fixture()
        remote.validate_condition_native(native, identity, condition)
        fields = ("status", "units", "duration_ms", "candidate_identity", "condition",
                  "unit_millisecond_counts_compared", "native_intervals_compared",
                  "native_terminal_footer_verified", "all_unit_event_counter_equality",
                  "production_episodes", "scientific_fidelity_qualified")
        for field in fields:
            changed = deepcopy(native)
            changed[field] = None
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "FULL_EPISODE_GATE_FAILED"):
                remote.validate_condition_native(changed, identity, condition)

    def test_exact_current_replay_counts_and_bits_are_required(self):
        native, identity, condition = self.native_fixture()
        for field in ("intervals_compared", "current_values_compared", "exact_float32_bits_equal",
                      "mismatched_values", "tolerance", "maximum_absolute_difference_pa"):
            changed = deepcopy(native)
            changed["independent_current_replay"][field] = None
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "FULL_EPISODE_GATE_FAILED"):
                remote.validate_condition_native(changed, identity, condition)

    def test_condition_binding_uses_existing_consumer(self):
        native, identity, condition = self.native_fixture()
        from population_dynamics import ca3_benchmark_readout
        index = identity["condition_index"]
        with patch.object(ca3_benchmark_readout, "bind_episode", return_value=dict(native=native, context={})):
            result = remote.bind_result(dict(self.job, condition_index=index), Path("/tmp/episode"))
        self.assertEqual(result["unit_millisecond_counts_verified"], 203408260)
        self.assertEqual(result["candidate_identity"], identity)
        self.assertEqual(result["production_episodes"], 0)

    def test_condition_failures_are_separate(self):
        import tempfile
        job = deepcopy(self.job)
        job["persistence_batch"]["conditions"] = job["persistence_batch"]["conditions"][:2]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            output = work / "output"
            output.mkdir()
            (work / "pack_results.py").write_text("# unit fixture\n")
            receipt = dict(execution_backend="merlin", merlin_job_run_id="fixture",
                           merlin_trial_id="fixture", hostname="fixture", persistence_prerequisite={})
            def persisted(out, root, target, item, **kwargs):
                item.update(persisted=True, checkpoint_path=target, result_sha256="fixture", result_bytes=1)
            with patch.object(remote, "resources", return_value={}), \
                 patch.object(remote.subprocess, "run"), \
                 patch.object(remote, "execute_source", side_effect=[RuntimeError("source failure"), None]), \
                 patch.object(remote, "bind_result", return_value={"production_episodes": 0}), \
                 patch.object(remote, "persist", side_effect=persisted):
                rows = remote.execute_condition_batch(job, work, output, "hdfs://fixture", receipt, {})
            self.assertEqual([row["exit_code"] for row in rows], [1, 0])
            self.assertEqual([row["source_pending_job"] for row in rows],
                             [row["source_job_id"] for row in job["persistence_batch"]["conditions"]])
            self.assertTrue(all(row["persisted"] for row in rows))

    def test_publication_failure_stops_following_conditions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            output = work / "output"
            output.mkdir()
            (work / "pack_results.py").write_text("# unit fixture\n")
            receipt = dict(execution_backend="merlin", merlin_job_run_id="fixture",
                           merlin_trial_id="fixture", hostname="fixture", persistence_prerequisite={})
            with patch.object(remote, "resources", return_value={}), \
                 patch.object(remote.subprocess, "run"), \
                 patch.object(remote, "execute_source") as execute, \
                 patch.object(remote, "bind_result", return_value={"production_episodes": 0}), \
                 patch.object(remote, "persist", side_effect=RuntimeError("publication failure")):
                with self.assertRaisesRegex(RuntimeError, "publication failure"):
                    remote.execute_condition_batch(self.job, work, output, "hdfs://fixture", receipt, {})
            self.assertEqual(execute.call_count, 1)
            self.assertFalse(receipt["condition_results"][0]["persisted"])

@unittest.skipUnless(os.environ.get("CA3_COMPLETION75_JOB"), "Requires the bound completion75 recovery inputs")
class TestCA3Completion75Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = json.loads(Path(os.environ["CA3_COMPLETION75_JOB"]).read_text())
        cls.job["candidate_execution"] = dict(source_repo=os.environ["CA3_COMPOSED_SOURCE"],
            packaged="/tmp/fixture_runtime", runtime_composition="/tmp/fixture_composition.json")

    @staticmethod
    def replace_document(binding, document):
        raw = json.dumps(document, sort_keys=True).encode()
        binding.update(content=raw.decode(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def test_recovery_command_identity_and_changed_cluster_branch(self):
        result = remote.validate_completion75_recovery(self.job)
        self.assertEqual(result["retry_branch"], "relevant_cluster_condition_changed")
        self.assertFalse(result["administrative_cancellation_cause_resolved"])
        self.assertEqual(result["historical_cancelled_jobs"], [609703, 609907])
        identity, condition = remote.persistence_condition_inputs(self.job, 4)
        self.assertEqual((identity["circuit_index"], identity["block_index"], identity["condition_index"]), (0, 0, 4))
        self.assertEqual(condition, dict(condition_id="completion_75", true_cue_units=413, false_cue_units=0))
        command = remote.source_command(self.job, Path("/tmp/episode"))
        self.assertEqual(command[command.index("--condition-index") + 1], "4")
        self.assertNotIn("--prefix-reference", command)
        self.assertIn("--current-replay-engine", command)
        self.assertIn("--runtime-composition", command)

    def test_both_cancelled_receipts_and_original_retry_branch_are_required(self):
        for index in (0, 1):
            job = deepcopy(self.job)
            binding = job["completion75_recovery"]["cancelled_attempts"][index]
            document = remote.bound_document(binding)
            document["terminal_report_exists"] = True
            self.replace_document(binding, document)
            with self.assertRaisesRegex(ValueError, "CANCELLED_ATTEMPT_BINDING_DIFFERS"):
                remote.validate_completion75_recovery(job)
        job = deepcopy(self.job)
        job["completion75_recovery"]["administrative_cancellation_cause_resolved"] = True
        with self.assertRaisesRegex(ValueError, "RECOVERY_SCOPE_DIFFERS"):
            remote.validate_completion75_recovery(job)
        job = deepcopy(self.job)
        binding = job["completion75_recovery"]["cancelled_attempts"][1]
        document = remote.bound_document(binding)
        document["retry_status"] = "UNCONDITIONAL_RETRY"
        self.replace_document(binding, document)
        with self.assertRaisesRegex(ValueError, "CANCELLED_ATTEMPT_BINDING_DIFFERS"):
            remote.validate_completion75_recovery(job)

    def test_recovery_rejects_other_conditions_and_pending_batch(self):
        for index in (3, 5, True, 10):
            job = deepcopy(self.job)
            job["condition_index"] = index
            with self.assertRaisesRegex(ValueError, "RECOVERY_SCOPE_DIFFERS"):
                remote.validate_completion75_recovery(job)
            with self.assertRaises(ValueError):
                remote.source_command(job, Path("/tmp/episode"))
        job = deepcopy(self.job)
        job["persistence_batch"] = {}
        with self.assertRaisesRegex(ValueError, "RECOVERY_SCOPE_DIFFERS"):
            remote.validate_completion75_recovery(job)

    def test_every_completed_condition_must_have_successful_persistent_source(self):
        for field, value in (("exit_code", 1), ("source_exit_code", 1), ("persisted", False)):
            job = deepcopy(self.job)
            recovery = job["completion75_recovery"]
            batch = remote.bound_document(recovery["merlin_batch_receipt"])
            rows = remote.bound_document(recovery["merlin_condition_results"])
            rows["conditions"][0][field] = value
            batch["condition_results"] = rows["conditions"]
            self.replace_document(recovery["merlin_batch_receipt"], batch)
            self.replace_document(recovery["merlin_condition_results"], rows)
            with self.assertRaisesRegex(ValueError, "CHANGED_CLUSTER_EXECUTION_REQUIRED"):
                remote.validate_completion75_recovery(job)

    def test_actual_platform_hardware_memory_and_source_are_required(self):
        for field, value in (("execution_backend", "slurm"), ("hostname", "gpu03"),
                             ("gpu_name", "NVIDIA L40S"), ("compute_capability", [8, 9])):
            job = deepcopy(self.job)
            recovery = job["completion75_recovery"]
            batch = remote.bound_document(recovery["merlin_batch_receipt"])
            batch[field] = value
            self.replace_document(recovery["merlin_batch_receipt"], batch)
            with self.assertRaisesRegex(ValueError, "CHANGED_CLUSTER_EXECUTION_REQUIRED"):
                remote.validate_completion75_recovery(job)
        for field in ("memory", "source"):
            job = deepcopy(self.job)
            recovery = job["completion75_recovery"]
            batch = remote.bound_document(recovery["merlin_batch_receipt"])
            if field == "memory":
                batch["resource_preflight"]["memory_evidence"]["platform_allocation"]["allocated_memory_mib"] = 1024
            else:
                batch["candidate_preflight"]["source_file_bindings"] = {}
            self.replace_document(recovery["merlin_batch_receipt"], batch)
            with self.assertRaisesRegex(ValueError, "CHANGED_CLUSTER_SOURCE_BINDING_DIFFERS"):
                remote.validate_completion75_recovery(job)

    def test_recovery_requires_the_existing_exact_prefix(self):
        job = deepcopy(self.job)
        job["completion75_recovery"]["prerequisite"]["origin"]["merlin_trial_id"] = "different"
        with self.assertRaisesRegex(ValueError, "REAL_MERLIN_PREFIX_PREREQUISITE_REQUIRED"):
            remote.validate_completion75_recovery(job)
        job = deepcopy(self.job)
        job["completion75_recovery"]["prerequisite"]["episode_report"]["content"] += " "
        with self.assertRaisesRegex(ValueError, "DOCUMENT_BINDING_DIFFERS"):
            remote.validate_completion75_recovery(job)

    def test_preflight_binds_recovery_and_rejects_previous_batch_proof(self):
        job = self.job
        code = Path(remote.__file__).read_text()
        proof = dict(status="PASS_CA3_COMPLETION75_RECOVERY_PREFLIGHT", exit_code=0, persisted=True,
            candidate_input_files=job["candidate_runtime"]["input_files"],
            source_file_bindings=job["execution_inputs"]["source_file_bindings"],
            runtime_build_sha256=job["execution_inputs"]["runtime_build_sha256"], image_meta=job["image_meta"],
            remote_sha256=hashlib.sha256(code.encode()).hexdigest(),
            completion75_recovery_sha256=remote.persistence_batch_fingerprint(job["completion75_recovery"]))
        args = (job["candidate_runtime"], proof, job["execution_inputs"], job["image_meta"], code)
        prepare.validate_candidate_preflight(*args, completion75_recovery=job["completion75_recovery"])
        for field, value in (("status", "PASS_CA3_PERSISTENCE_CONDITION_PREFLIGHT"),
                             ("completion75_recovery_sha256", "0" * 64)):
            changed = deepcopy(proof)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "MATCHING_CPU_CANDIDATE_PREFLIGHT_REQUIRED"):
                prepare.validate_candidate_preflight(args[0], changed, *args[2:],
                    completion75_recovery=job["completion75_recovery"])

    def test_recovery_uses_archive_binding_and_exact_full_episode_gate(self):
        from population_dynamics import ca3_benchmark_readout
        native = remote.bound_document(self.job["completion75_recovery"]["prerequisite"]["episode_report"])
        identity, condition = remote.persistence_condition_inputs(self.job, 4)
        native.update(candidate_identity=identity, condition=condition)
        bound = dict(native=native, context={})
        with patch.object(ca3_benchmark_readout, "bind_episode", return_value=bound):
            result = remote.bind_result(self.job, Path("/tmp/episode"))
        self.assertEqual(result["unit_millisecond_counts_verified"], 203408260)
        self.assertEqual(result["candidate_identity"], identity)
        for field in ("native_terminal_footer_verified", "unit_millisecond_counts_compared"):
            changed = deepcopy(bound)
            changed["native"][field] = None
            with patch.object(ca3_benchmark_readout, "bind_episode", return_value=changed):
                with self.assertRaisesRegex(ValueError, "FULL_EPISODE_GATE_FAILED"):
                    remote.bind_result(self.job, Path("/tmp/episode"))
        with patch.object(ca3_benchmark_readout, "bind_episode", side_effect=ValueError("archive identity differs")):
            with self.assertRaisesRegex(ValueError, "archive identity differs"):
                remote.bind_result(self.job, Path("/tmp/episode"))

    def run_main_fixture(self, *, resource_failure=False):
        job = deepcopy(self.job)
        job.update(candidate_preflight={}, expected_compute_capability=[8, 0])
        job["resource_shape"].update(gpu=1, cpu=15)
        saved = []
        def published(output, work, target, receipt, **kwargs):
            receipt.update(persisted=True, checkpoint_path=target)
            saved.append(deepcopy(receipt))
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda _: (8, 0),
                               get_device_name=lambda _: "NVIDIA A800-SXM4-80GB")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            path = work / "job.json"
            path.write_text(json.dumps(job))
            with patch.object(sys, "argv", ["ca3_merlin_remote", "--job-json", str(path)]), \
                 patch.dict(os.environ, RH2_JOB_RUN_ID="fixture-run", ARNOLD_TRIAL_ID="fixture-trial",
                            RH2_CHECKPOINT_HDFS="hdfs://fixture", ARNOLD_APPLIED_GPU_NUM="1", CUDA_VISIBLE_DEVICES="0"), \
                 patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)), \
                 patch.object(remote.socket, "gethostname", return_value="fixture-merlin"), \
                 patch.object(remote.signal, "signal"), \
                 patch.object(remote.subprocess, "run"), patch.object(remote, "hdfs_put"), \
                 patch.object(remote.importlib.metadata, "version", return_value="fixture"), \
                 patch.object(remote, "resources", side_effect=RuntimeError("fixture resource failure") if resource_failure else None,
                              return_value=dict(memory_evidence=dict(admission=dict(status="PASS")))), \
                 patch.object(remote, "download_inputs", return_value=Path("/tmp/unused")), \
                 patch.object(remote, "stage_inputs"), patch.object(remote, "verify_scientific_inputs"), \
                 patch.object(remote, "compose_candidate", return_value=job["candidate_execution"]), \
                 patch.object(remote, "execute_source") as execute, patch.object(remote, "execute_condition_batch") as batch, \
                 patch.object(remote, "bind_result", return_value=dict(production_episodes=0, scientific_fidelity_qualified=False)), \
                 patch.object(remote, "persist", side_effect=published), patch("builtins.print"):
                code = remote.main()
                return code, saved, execute.call_args_list, batch.call_count

    def test_recovery_failure_keeps_namespace_history_and_persistence(self):
        code, saved, commands, batches = self.run_main_fixture(resource_failure=True)
        self.assertEqual(code, 1)
        self.assertFalse(commands)
        self.assertEqual(batches, 0)
        self.assertEqual(saved[0]["status"], "FAILED_CA3_MERLIN_COMPLETION75_RECOVERY")
        self.assertEqual(saved[0]["checkpoint_path"], "hdfs://fixture/ca3/completion75_recovery/fixture-trial")
        self.assertFalse(saved[0]["administrative_cancellation_cause_resolved"])
        self.assertNotIn("source_pending_job", saved[0])

    def test_recovery_success_executes_only_one_native_condition(self):
        code, saved, commands, batches = self.run_main_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 1)
        self.assertEqual(batches, 0)
        command = commands[0].args[0]
        self.assertEqual(command[command.index("--condition-index") + 1], "4")
        self.assertNotIn("--prefix-reference", command)
        self.assertEqual(saved[0]["status"], "COMPLETE_CA3_MERLIN_COMPLETION75_RECOVERY")
        self.assertEqual(saved[0]["historical_cancelled_jobs"], [609703, 609907])
        self.assertNotIn("source_pending_job", saved[0])
        self.assertFalse(saved[0]["scientific_fidelity_qualified"])


if __name__ == "__main__":
    unittest.main()
