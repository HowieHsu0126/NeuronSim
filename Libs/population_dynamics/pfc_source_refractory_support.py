"""Validate one authenticated source episode's refractory support; no replay."""
from __future__ import annotations
import hashlib
import json
import re
from copy import deepcopy

EXECUTOR_SHA256 = "aa1d2ccca0dab4621bf0b2e99f0ea75e2e037e69c8a31d52877ae92a5c794171"
SOURCE_ENTRY_SHA256 = "416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720"
POPULATIONS = {"E": (80000,160), "I": (20000,80)}


def require(value, message):
    if not value:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def document(data, role):
    require(isinstance(data,bytes), "BYTE_INPUT_REQUIRED:"+role)
    try:
        value=json.loads(data)
    except (UnicodeDecodeError,json.JSONDecodeError) as error:
        raise ValueError("JSON_INPUT_INVALID:"+role) from error
    require(isinstance(value,dict), "JSON_OBJECT_REQUIRED:"+role)
    return value,dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def delivered(delivery, path, binding):
    rows=delivery.get("files")
    require(isinstance(rows,list) and all(isinstance(row,dict) for row in rows), "DELIVERY_FILES_REQUIRED")
    selected=[row for row in rows if row.get("path")==path]
    require(len(selected)==1 and type(selected[0].get("bytes")) is int
            and selected[0]["bytes"]==binding["bytes"] and selected[0].get("sha256")==binding["sha256"],
            "DELIVERED_FILE_BINDING_CHANGED:"+path)
    return deepcopy(selected[0])


def flat_source_config(config):
    require(isinstance(config,dict), "NATIVE_RESOLVED_CONFIG_REQUIRED")
    if "source_config" in config:
        require(isinstance(config["source_config"],dict), "UNSUPPORTED_NATIVE_SOURCE_CONFIG_SHAPE")
        config=config["source_config"]
    metadata={"cases","initial_rng_seed","external_noise_seed","source_rng_seed"}
    result={key:value for key,value in config.items() if key not in metadata and not key.startswith("benchmark_")}
    require(result and all(not isinstance(value,(dict,list)) for value in result.values()),
            "UNSUPPORTED_NATIVE_SOURCE_CONFIG_SHAPE")
    return result


def validate_episode(expected, readout_report_bytes, source_delivery_bytes, source_report_bytes,
                     qualification_delivery_bytes, qualification_proof_bytes):
    """Root authenticates snapshot/phase bytes; this validates the linked JSON chain."""
    require(isinstance(expected,dict) and isinstance(expected.get("identity"),dict)
            and valid_hash(expected.get("config_sha256")), "FROZEN_EPISODE_EXPECTATION_REQUIRED")
    identity=expected["identity"];episode=identity.get("episode_id")
    require(isinstance(episode,str) and episode and type(identity.get("initial_state_seed")) is int
            and type(identity.get("background_seed")) is int, "FROZEN_EPISODE_IDENTITY_REQUIRED")
    readout,rb=document(readout_report_bytes,"readout")
    delivery,db=document(source_delivery_bytes,"source_delivery")
    native,nb=document(source_report_bytes,"source_report")
    qdelivery,qdb=document(qualification_delivery_bytes,"qualification_delivery")
    proof,qb=document(qualification_proof_bytes,"qualification")
    require(readout.get("identity")==identity and readout.get("config_sha256")==expected["config_sha256"],
            "READOUT_EPISODE_IDENTITY_CHANGED")
    require(readout.get("source_delivery_sha256")==db["sha256"]
            and readout.get("source_report_sha256")==nb["sha256"], "READOUT_SOURCE_BINDING_CHANGED")
    require(delivery.get("status")=="ARCHIVED_PFC_SOURCE_EXECUTION"
            and delivery.get("result_status")=="PFC_SOURCE_EXECUTION_COMPLETE"
            and delivery.get("source_execution_complete") is True and delivery.get("case_id")==episode
            and type(delivery.get("exit_code")) is int and delivery["exit_code"]==0,
            "COMPLETE_SOURCE_DELIVERY_REQUIRED")
    source_binding=delivered(delivery,"source_run/report.json",nb)
    require(qdelivery.get("status")=="ARCHIVED_PFC_STREAM_VALIDATION"
            and qdelivery.get("result_status")=="PASS_NATIVE_STREAM_SOURCE_EXECUTION",
            "QUALIFICATION_DELIVERY_REQUIRED")
    qualification_binding=delivered(qdelivery,"comparison/qualification.json",qb)
    qualification=native.get("backend_qualification",{})
    require(isinstance(qualification,dict) and qualification.get("delivery_sha256")==qdb["sha256"]
            and qualification.get("receipt_sha256")==qb["sha256"]
            and qualification.get("receipt_file")=="comparison/qualification.json", "NATIVE_QUALIFICATION_BINDING_CHANGED")
    if "qualification" in qualification:
        require(qualification["qualification"]==proof, "EMBEDDED_QUALIFICATION_CHANGED")
    signature=expected.get("source_signature")
    require(isinstance(signature,dict) and set(signature)=={"source_config","case"}
            and isinstance(signature["source_config"],dict) and isinstance(signature["case"],dict),
            "UNSUPPORTED_FROZEN_SOURCE_SIGNATURE")
    source=flat_source_config(native.get("resolved_config"))
    require(source==signature["source_config"], "SOURCE_PARAMETER_SIGNATURE_CHANGED")
    case_part=signature["case"]
    require(set(case_part)=={"random_stream_policy","stimulus_strength_nA","distractor_strength_nA"},
            "UNSUPPORTED_FROZEN_CASE_SIGNATURE")
    case=dict(case_part,case_id=episode,initial_rng_seed=identity["initial_state_seed"],external_noise_seed=identity["background_seed"])
    require(native.get("status")=="PFC_SOURCE_EXECUTION_COMPLETE" and native.get("source_execution_complete") is True
            and native.get("case_id")==episode and native.get("selected_case")==case
            and native.get("config_sha256")==expected["config_sha256"], "SOURCE_CASE_OR_CONFIG_CHANGED")
    require(native.get("registered_case_ids")==[episode]
            and all(type(native.get(key)) is int and native[key]==1 for key in
                ("registered_case_count","selected_case_count","completed_case_count")), "SINGLE_COMPLETE_SOURCE_CASE_REQUIRED")
    require(source.get("scope")=="source_reproduction" and source.get("excitatory_neurons")==80000
            and source.get("inhibitory_neurons")==20000 and source.get("dt_ms")==0.0125
            and source.get("duration_ms")==5000 and case_part["random_stream_policy"]=="independent_initial_external",
            "FROZEN_FULL_SIZE_SOURCE_REQUIRED")
    dimensions=native.get("dimensions",{});driver=native.get("driver",{});inputs=native.get("inputs_used",{})
    require(isinstance(dimensions,dict) and isinstance(driver,dict) and isinstance(inputs,dict), "SOURCE_DIMENSIONS_DRIVER_REQUIRED")
    for field,value in (("n_e",80000),("n_i",20000),("coupled_core_units",100000),("duration_ticks",400000)):
        require(type(dimensions.get(field)) is int and dimensions[field]==value, "SOURCE_DIMENSION_CHANGED:"+field)
    require(dimensions.get("dt_seconds")==0.0000125 and dimensions.get("duration_seconds")==5
            and native.get("completed_duration_ticks")==400000, "SOURCE_CLOCK_OR_DURATION_CHANGED")
    code=native.get("code",{})
    require(isinstance(code,dict) and code.get("executor_sha256")==proof.get("executor_sha256")==EXECUTOR_SHA256
            and native.get("source_entry_sha256")==proof.get("source_entry_sha256")==SOURCE_ENTRY_SHA256,
            "FROZEN_SOURCE_CODE_CHANGED")
    require(valid_hash(proof.get("drive_sha256")) and code.get("drive_sha256")==proof["drive_sha256"],
            "FROZEN_DRIVER_BINDING_CHANGED")
    require(proof.get("status")=="PASS_NATIVE_STREAM_SOURCE_EXECUTION" and proof.get("backend")==native.get("backend")=="numpy"
            and proof.get("validated_dt_ms")==[0.0125] and proof.get("numpy_version")==driver.get("numpy_version")=="2.4.2",
            "QUALIFIED_REFERENCE_CLOCK_REQUIRED")
    validations=native.get("validations",{});consumed=validations.get("complete_source_input_consumption",{})
    require(isinstance(validations,dict) and isinstance(consumed,dict) and consumed.get("passed") is True
            and driver.get("source_entry_sha256")==SOURCE_ENTRY_SHA256
            and all(type(driver.get(key)) is int and driver[key]==400000 for key in ("total_ticks","ticks_generated"))
            and driver.get("dt_seconds")==0.0000125 and driver.get("random_stream_policy")=="independent_initial_external"
            and driver.get("seeds")==dict(initial_rng_seed=identity["initial_state_seed"],external_noise_seed=identity["background_seed"])
            and inputs.get("ticks_consumed")==400000 and consumed.get("expected_ticks")==consumed.get("generated_ticks")==400000,
            "FULL_NATIVE_INPUT_CONSUMPTION_REQUIRED")
    summaries={}
    for population,(units,required) in POPULATIONS.items():
        require(driver.get("n_e" if population=="E" else "n_i")==units
                and inputs.get(population,{}).get("values")==units*400000
                and consumed.get("expected_values",{}).get(population)==units*400000,
                "FULL_POPULATION_INPUT_CONSUMPTION_CHANGED:"+population)
        events=inputs[population].get("events")
        require(type(events) is int and events>=0 and driver.get("external_event_totals",{}).get(population)==events
                and consumed.get("event_totals",{}).get(population)==events
                and native.get("source_external_event_totals",{}).get(population)==events,
                "SOURCE_EXTERNAL_EVENT_TOTAL_CHANGED:"+population)
        spacing=validations.get("refractory_spacing_"+population)
        require(isinstance(spacing,dict) and spacing.get("passed") is True
                and type(spacing.get("minimum_required_ticks")) is int and spacing["minimum_required_ticks"]==required
                and type(spacing.get("violation_count")) is int and spacing["violation_count"]==0
                and type(spacing.get("within_unit_event_pairs")) is int and spacing["within_unit_event_pairs"]>=0,
                "REFRACTORY_SUPPORT_INVALID:"+population)
        pairs=spacing["within_unit_event_pairs"];minimum=spacing.get("minimum_observed_ticks")
        require((pairs==0 and minimum is None) or (pairs>0 and type(minimum) is int and minimum>=required),
                "REFRACTORY_MINIMUM_INCONSISTENT:"+population)
        summaries[population]=dict(units=units,**deepcopy(spacing))
    for value in (native,delivery,qdelivery,proof):
        require(type(value.get("production_episodes")) is int and value["production_episodes"]==0
                and value.get("scientific_fidelity_qualified") is False, "SOURCE_SCOPE_PROMOTED")
    return dict(status="VALIDATED_EPISODE_REFRACTORY_SUPPORT",identity=deepcopy(identity),config_sha256=expected["config_sha256"],
        source_signature=deepcopy(signature),source_archive=readout.get("source_archive"),populations=summaries,
        executor_sha256=EXECUTOR_SHA256,source_entry_sha256=SOURCE_ENTRY_SHA256,drive_sha256=proof["drive_sha256"],
        bindings=dict(readout_report=rb,source_delivery=db,source_report=source_binding,qualification_delivery=qdb,qualification=qualification_binding),
        silence_interpretation="No population-silence inference from zero within-unit event pairs",
        source_simulations=0,production_episodes=0,source_qualification_granted=False,scientific_parameter_domain_qualified=False,
        scientific_claim_qualified=False,scaling_qualified=False,uncertainty_coverage_qualified=False)
