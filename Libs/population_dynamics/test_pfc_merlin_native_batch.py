"""Remaining native conditions require the complete persisted canary PASS."""
import copy

import pytest

from .pfc_merlin_native_batch import validate_canary, validate_request


def request():
    return dict(schema="pfc_merlin_native_conditions_request_v1", candidate_source_run="6de397acc866aadd",
        candidate_source_trial="435906314", canary_run="58d9076a85680612", canary_trial="437239519",
        canary_checkpoint="hdfs://checkpoint/58d9076a85680612/pfc_native_correspondence",
        canary_candidate_source_report_sha256="report", canary_consumer_sha256="software",
        tasks=[dict(index=i, identity=dict(circuit_index=23, block_index=6, condition_index=i-690,
            episode_id=f"pfc_working_memory_circuit_023_block_006_condition_{i-690:02d}", split="train"))
            for i in range(691, 700)])


def test_exact_remaining_nine_excludes_canary_and_duplicates():
    value = request(); validate_request(value)
    for indices in ([690, *range(692, 700)], [691, *range(691, 699)], list(range(691, 699))):
        changed = copy.deepcopy(value)
        changed["tasks"] = [dict(index=i, identity={}) for i in indices]
        with pytest.raises(ValueError, match="EXACT_REMAINING_NINE"):
            validate_request(changed)


def test_changed_condition_or_canary_identity_rejects():
    value = request(); value["tasks"][0]["identity"]["block_index"] = 7
    with pytest.raises(ValueError, match="CONDITION_IDENTITY_CHANGED"):
        validate_request(value)
    value = request(); value["canary_trial"] = "different"
    with pytest.raises(ValueError, match="CANARY_BINDING_CHANGED"):
        validate_request(value)


def canary():
    comparison = dict(status="PASS_NATIVE_CAMPAIGN_CORRESPONDENCE",
        identity=dict(episode_id="pfc_working_memory_circuit_023_block_006_condition_00"),
        execution=dict(run_id="58d9076a85680612", trial_id="437239519"),
        native_receipt_sha256="receipt", candidate_source_report_sha256="report",
        full_scale_source_case_complete=True, candidate_source_simulations=0, new_native_source_simulations=1,
        production_episodes=0, source_qualification_granted=False, scaling_qualified=False,
        native_bundle_uri="hdfs://native", array_comparisons=2, passed_arrays=2,
        arrays=[dict(array="spike_ticks", exact=True, passed=True),
                dict(array="terminal_voltage_mV", atol=1e-10, rtol=0., passed=True)])
    receipt = dict(status="PERSISTED_COMPLETE_NATIVE_EXECUTION", exit_code=0,
        run_id="58d9076a85680612", trial_id="437239519", bundle_uri="hdfs://native")
    validation = dict(exit_code=0, run_id="58d9076a85680612", trial_id="437239519",
        files={"Libs/population_dynamics/pfc_merlin_native_correspondence.py": "software"})
    return comparison, receipt, validation


def test_native_completion_without_comparison_pass_is_insufficient():
    comparison, receipt, validation = canary()
    validate_canary(comparison, receipt, validation, request(), "receipt")
    comparison["status"] = "FAILED_NATIVE_CAMPAIGN_CORRESPONDENCE"
    with pytest.raises(ValueError, match="COMPLETE_COMPARISON_PASS_REQUIRED"):
        validate_canary(comparison, receipt, validation, request(), "receipt")


def test_changed_software_or_tolerance_blocks_batch():
    comparison, receipt, validation = canary()
    comparison["arrays"][1]["atol"] = 1e-5
    with pytest.raises(ValueError, match="ARRAY_CRITERIA_CHANGED"):
        validate_canary(comparison, receipt, validation, request(), "receipt")
    comparison, receipt, validation = canary()
    validation["files"]["Libs/population_dynamics/pfc_merlin_native_correspondence.py"] = "different"
    with pytest.raises(ValueError, match="SOFTWARE_BINDING_CHANGED"):
        validate_canary(comparison, receipt, validation, request(), "receipt")
