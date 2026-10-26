"""Source-bound RNG boundary and actual compiled host-stream checks."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .ca3_realization import (SourceRealization, contract, from_record, program,
                              source_reset_contract, validate_source_reset_contract,
                              RESET_KERNEL_PATH, RESET_KERNEL_SHA256, SOURCE_RESET_VR)
from . import ca3_host_current_replay as replay
from .ca3_reference import PROJECT, MAIN, RETRIEVAL_PROJECT, RETRIEVAL_MAIN
from .test_ca3_host_current_replay import engine, sample, intervals
from .test_ca3_source_dataset import source, realized_copy, delivery
from .test_ca3_native_observation_replay import refresh_bindings
from . import ca3_source_dataset as exporter
from .provenance import canonical_json, sha256_file

SPEC = SourceRealization("coupled-development", "noise-01", "retrieval",
                         "development", 10, 1001, 2001)
AUTHOR = Path(__file__).resolve().parents[1] / "external/ca3_formation_retrieval"


@pytest.mark.parametrize("seed", [0, -1, 2147483647, 1.5, True])
def test_seed_alias_and_conversion_rejected(seed):
    with pytest.raises(ValueError, match="SEED_INVALID"):
        replace(SPEC, engine_seed=seed)


def test_explicit_identity_and_qualification_boundary():
    assert from_record(SPEC.record()) == SPEC
    with pytest.raises(ValueError, match="SCHEMA_INVALID"):
        from_record(dict(SPEC.record(), unexpected=True))
    with pytest.raises(ValueError, match="QUALIFICATION_SPLIT"):
        replace(SPEC, split="test")
    result = contract(SPEC)
    assert result["initial_state_policy"] == "fresh_source_reset_not_checkpoint_state"
    assert result["common_seed_for_nine_host_engines"]
    assert not result["source_topology_noise_coupling_preserved"]
    assert not result["independent_noise_qualified"]


@pytest.mark.parametrize("project,main,phase", [
    (PROJECT, MAIN, "formation"), (RETRIEVAL_PROJECT, RETRIEVAL_MAIN, "retrieval")])
def test_actual_author_source_boundary_and_byte_preservation(project, main, phase):
    original = (AUTHOR/project/main).read_bytes()
    spec = (replace(SPEC, phase=phase, circuit_seed=1031, uniform_seed=1031, engine_seed=1)
            if phase == "formation" else replace(SPEC, circuit_seed=1031))
    changed = program(original, spec)
    needle = b"sim.setupNetwork();"
    before, after = original.split(needle)
    new_before, new_after = changed.split(needle)
    assert new_before == before.replace(b"int randSeed = 10;", b"int randSeed = 1031;", 1)
    assert new_after.endswith(after)
    injection = new_after[:-len(after)]
    if phase == "formation":
        assert injection == b""
        assert contract(spec)["source_topology_noise_coupling_preserved"]
    else:
        assert injection.count(b".seed(2001U);") == 9
        assert injection.count(b"srand48(1001L);") == 1
    assert b"srand(" not in injection
    assert original == (AUTHOR/project/main).read_bytes()
    with pytest.raises(ValueError, match="SOURCE_RNG_BOUNDARY_CHANGED"):
        program(changed, spec)
    early = original.replace(needle, b"pyr_generator(); " + needle, 1)
    with pytest.raises(ValueError, match="ENGINE_USED_BEFORE_BOUNDARY"):
        program(early, spec)


def _values(engine, groups, commands, spec):
    result = subprocess.run([str(engine), *spec.replay_arguments()],
                            input=replay.engine_input(groups, commands),
                            capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4").reshape(1001, -1).copy()


def test_compiled_explicit_stream_reproducibility_and_noise_separation(engine, sample):
    groups, commands, historical = sample
    values = _values(engine, groups, commands, SPEC)
    np.testing.assert_array_equal(values.view("<u4"), _values(engine, groups, commands, SPEC).view("<u4"))
    for changed in (replace(SPEC, uniform_seed=1002), replace(SPEC, engine_seed=2002)):
        assert not np.array_equal(values, _values(engine, groups, commands, changed))
    assert not np.array_equal(values, historical)
    checked = replay.compare_currents(engine, groups, commands, intervals(values), realization=SPEC)
    assert checked["exact_float32_bits_equal"]
    assert checked["current_values_compared"] == values.size
    assert checked["tolerance"] == 0
    pyr = next(g for g in groups if g.name == replay.PYRAMIDAL)
    for start in (520, 750, 980):
        for g in groups:
            if g.name != replay.PYRAMIDAL:
                np.testing.assert_array_equal(values[start:start+20, g.start:g.stop],
                    np.broadcast_to(values[start-1, g.start:g.stop], (20, g.units)))
    for t, unit in zip(commands["time_ms"], commands["unit_ids"]):
        assert values[t, unit] == np.float32(10000)
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="RETRIEVAL_REALIZATION"):
        replay.compare_currents(engine, groups, commands, intervals(values),
                                realization=replace(SPEC, phase="formation", uniform_seed=10, engine_seed=1))


def test_compiled_seed_boundary_identity_and_rejection(engine, sample):
    source = json.loads(subprocess.check_output([str(engine), "--identity"], text=True))
    explicit = json.loads(subprocess.check_output(
        [str(engine), *SPEC.replay_arguments(), "--identity"], text=True))
    assert not source["explicit_noise_block"]
    assert source["post_seed_discard_draws_per_unit"] == 9
    assert explicit["explicit_noise_block"]
    assert explicit["uniform_seed"] == 1001
    assert explicit["common_engine_seed"] == 2001
    assert explicit["post_seed_discard_draws_per_unit"] == 0
    groups, commands, _ = sample
    for bad in ("0", "2147483647", "-1", "1x"):
        result = subprocess.run([str(engine), "--noise-block", bad, "2001"],
                                input=replay.engine_input(groups, commands), capture_output=True)
        assert result.returncode != 0
        assert not result.stdout


def test_declared_new_circuit_seed_in_host_replay_identity(engine, sample):
    groups, commands, _ = sample
    changed = replace(SPEC, circuit_seed=11)
    identity = json.loads(subprocess.check_output(
        [str(engine), *changed.replay_arguments(), "--identity"], text=True))
    assert identity["carlsim_seed"] == 11
    assert identity["uniform_seed"] == 1001 and identity["common_engine_seed"] == 2001
    # Explicit retrieval resets erase the earlier uniform state. This equality
    # does not say that the separately constructed graph or source reset is equal.
    np.testing.assert_array_equal(_values(engine, groups, commands, changed),
                                  _values(engine, groups, commands, SPEC))


def test_formation_configuration_retains_source_noise_coupling():
    with pytest.raises(ValueError, match="SOURCE_COUPLED_UNIFORM_AND_DEFAULT_ENGINES"):
        replace(SPEC, phase="formation", circuit_seed=11)
    spec = replace(SPEC, phase="formation", circuit_seed=11, uniform_seed=11, engine_seed=1)
    assert contract(spec)["source_topology_noise_coupling_preserved"] is True
    assert "before_setup" in contract(spec)["noise_seed_boundary"]


@pytest.fixture
def reset_kernel():
    path = Path(__file__).resolve().parents[1] / "external/carlsim_ca3" / RESET_KERNEL_PATH
    return path.read_bytes()


@pytest.mark.parametrize("project", [PROJECT, RETRIEVAL_PROJECT])
def test_native_reset_authenticates_both_source_headers(project, reset_kernel):
    header = (AUTHOR/project/"generateCONFIGStateSTP.h").read_bytes()
    result = source_reset_contract(header, reset_kernel)
    assert result["source_header_sha256"] == hashlib.sha256(header).hexdigest()
    assert result["reset_kernel"] == dict(path=RESET_KERNEL_PATH, sha256=RESET_KERNEL_SHA256)
    assert result["scope"] == ["voltage", "recovery", "Izh_ref_c"]
    assert result["parameter_standard_deviations_zero"]
    assert "nine draws per neuron even when sd is zero" in result["parameter_draw_rule"]
    assert [row["population"] for row in result["populations"]] == [name for name, _ in SOURCE_RESET_VR]
    for row in result["populations"]:
        assert row["voltage_mV"] == float(np.float32(row["source_vr_literal"]))
        assert row["recovery"] == 0.0 and row["Izh_ref_c"] == 0
    assert result["native_reset_deterministic"]
    for name in ("noise_block_changes_initial_state", "complete_microscopic_state_qualified",
                 "independent_microscopic_states_qualified"):
        assert result[name] is False
    # Commented alternatives must remain inert. Raw header bytes remain bound.
    commented = header + b"\n// sim.setNeuronParameters(fake, 1, 1);\n"
    assert source_reset_contract(commented, reset_kernel)["populations"] == result["populations"]
    assert source_reset_contract(commented, reset_kernel)["source_header_sha256"] != result["source_header_sha256"]


def test_native_reset_rejects_kernel_and_active_parameter_changes(reset_kernel):
    header = (AUTHOR/PROJECT/"generateCONFIGStateSTP.h").read_bytes()
    with pytest.raises(ValueError, match="PINNED_KERNEL_REQUIRED"):
        source_reset_contract(header, reset_kernel+b"\n")
    anchor = b"sim.setNeuronParameters(CA3_QuadD_LM, 186.0, 0.0"
    assert header.count(anchor) == 1
    with pytest.raises(ValueError, match="NONZERO_PARAMETER_SD:CA3_QuadD_LM:C"):
        source_reset_contract(header.replace(anchor, anchor[:-3]+b"0.1", 1), reset_kernel)
    with pytest.raises(ValueError, match="PINNED_PARAMETER_CALLS_REQUIRED"):
        source_reset_contract(header.replace(b"-73.4821116922868", b"-73.0", 1), reset_kernel)
    with pytest.raises(ValueError, match="EIGHT_NINE_PARAMETER_CALLS_REQUIRED"):
        source_reset_contract(header+b"\nsim.setNeuronParameters(fake, 1, 1);\n", reset_kernel)


def test_reset_declaration_is_optional_and_bound_to_existing_header_receipt(reset_kernel):
    header = (AUTHOR/PROJECT/"generateCONFIGStateSTP.h").read_bytes()
    declaration = source_reset_contract(header, reset_kernel)
    legacy = dict(source_realization=contract(SPEC))
    historical = deepcopy(legacy)
    assert validate_source_reset_contract(legacy) is None
    assert legacy == historical
    report = dict(legacy, source_initial_state_contract=declaration,
        source_file_byte_checks=[dict(path="generateCONFIGStateSTP.h",
                                     staged_sha256=declaration["source_header_sha256"])])
    assert validate_source_reset_contract(report) == declaration
    validated = validate_source_reset_contract(report)
    validated["populations"][0]["voltage_mV"] = 0
    assert report["source_initial_state_contract"] == declaration
    missing = deepcopy(report); del missing["source_realization"]
    with pytest.raises(ValueError, match="EXPLICIT_REALIZATION_REQUIRED"):
        validate_source_reset_contract(missing)
    for field in ("source_header_sha256", "noise_block_changes_initial_state", "populations"):
        changed = deepcopy(report)
        if field == "source_header_sha256":
            changed["source_initial_state_contract"][field] = "0"*64
        elif field == "populations":
            changed["source_initial_state_contract"][field][0]["voltage_mV"] = 0
        else:
            changed["source_initial_state_contract"][field] = True
        with pytest.raises(ValueError, match="SOURCE_CONTRACT_BINDING_DIFFERS"):
            validate_source_reset_contract(changed)
    duplicate = deepcopy(report)
    duplicate["source_file_byte_checks"] *= 2
    with pytest.raises(ValueError, match="SOURCE_CONTRACT_BINDING_DIFFERS"):
        validate_source_reset_contract(duplicate)


def test_dataset_propagates_reset_without_changing_existing_identities(source, reset_kernel, tmp_path):
    import h5py

    reference, _, legacy_native, groups, *_ = source
    primary, analysis, config, spec = realized_copy(source, tmp_path/"realized")
    before = json.loads((primary/"native_report.json").read_text())
    units = exporter.source_units(before, groups, realization_source=primary/"realization_source.cpp",
                                  reference_native=legacy_native)
    parameters = exporter._source_parameters(before, include_config_path=False)
    condition = dict(circuit_id=units.circuit_id, pattern_sha256=before["pattern_sha256"],
        seeds=dict(source_carlsim_seed=10, source_host_srand_seed=3,
                   source_uniform_seed=spec.uniform_seed, source_engine_seed=spec.engine_seed),
        source_realization=before["source_realization"])
    expected_key = hashlib.sha256(canonical_json(condition).encode()).hexdigest()[:20]
    declaration = source_reset_contract((primary/"config.h").read_bytes(), reset_kernel)
    for name in ("build_report.json", "native_report.json"):
        path = primary/name; report = json.loads(path.read_text())
        report["source_initial_state_contract"] = declaration
        assert validate_source_reset_contract(report) == declaration
        path.write_text(json.dumps(report))
    refresh_bindings(primary)
    native = json.loads((primary/"native_report.json").read_text())
    assert exporter._source_parameters(native, include_config_path=False) == parameters
    after_units = exporter.source_units(native, groups,
        realization_source=primary/"realization_source.cpp", reference_native=legacy_native)
    assert after_units.circuit_id == units.circuit_id
    assert after_units.circuit_provenance == units.circuit_provenance
    path = analysis/"reconciliation.json"; proof = json.loads(path.read_text())
    proof.update(original_report_sha256=sha256_file(primary/"native_report.json"),
                 original_recording_sha256=sha256_file(primary/"observations/recording.json"))
    path.write_text(json.dumps(proof))
    path = analysis/"control/report.json"; delivered = json.loads(path.read_text())
    delivered["input_sha256"]["reconciliation"] = sha256_file(analysis/"reconciliation.json")
    path.write_text(json.dumps(delivered))
    delivery(primary, "COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED")
    delivery(analysis, "PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL")
    output = tmp_path/"export"
    report = exporter.export(primary, analysis, output, source_realization_config=config,
                             circuit_reference_primary=reference)
    assert report["episode_id"] == "ca3_source_"+expected_key
    assert report["block_id"] == "ca3_source_block_"+expected_key
    assert report["verification"]["all_unit_bin_values_checked"] == 168274106
    assert report["production_episodes"] == 0 and not report["scientific_fidelity_qualified"]
    dataset = exporter.CircuitDataset(output/"dataset", splits=("development",))
    assert dataset.catalog["binding"]["source_initial_state_contract"] == declaration
    row = dataset.episodes[report["episode_id"]]; shard = dataset.shards[row["shard"]]
    with h5py.File(dataset._verified(shard["path"], shard["sha256"]), "r") as archive:
        record = json.loads(archive[report["episode_id"]].attrs["metadata_json"])["source_record"]
    assert record["source_initial_state_contract"] == declaration
    assert record["source_realization"] == contract(spec)
    assert record["source_noise_block_scope"] == "post-setup external noise; deterministic reset is not initial-state diversity"
    assert not record["independent_microscopic_states_qualified"]
    # A new native declaration requires the same declaration in its build.
    path = primary/"build_report.json"; build = json.loads(path.read_text())
    del build["source_initial_state_contract"]; path.write_text(json.dumps(build))
    delivery(primary, "COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED")
    rejected = tmp_path/"rejected"
    with pytest.raises(ValueError, match="INITIAL_STATE_BUILD_BINDING_CHANGED"):
        exporter.export(primary, analysis, rejected, source_realization_config=config,
                        circuit_reference_primary=reference)
    assert not rejected.exists()
