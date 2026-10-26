"""Tiny software fixtures exercise readout contracts, never scientific coverage."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from . import v1_direction_response as response
from .provenance import sha256_file
from .v1_source_dataset import input_plan_identity


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    # These reduced inventories exist only in software fixtures; production has no override.
    monkeypatch.setattr(response, "CORTICAL_UNITS", 3)
    monkeypatch.setattr(response, "LGN_UNITS", 2)
    configs = tmp_path / "Libs/configs"; configs.mkdir(parents=True)
    protocol = dict(native_resolution_ms=.25, cortical_units=3, new_directions_deg=list(response.DIRECTIONS[:-1]),
        reused_direction_deg=315, maximum_new_full_conditions=7, trials_per_direction=1,
        vary_only="stimulus.direction_deg", noise_pairing_claim="none", source_fidelity_qualified=False,
        production_eligible=False, formal_protocol_frozen=False, visual_config_base="Libs/configs/visual.yaml",
        references=dict(reused_native="native/run_900", reused_observation="observation", reused_controls="controls"),
        execution=dict(full_lgn_units=2), readout=dict(retain_all_cortical_units=True, retain_silent_units=True,
            retain_source_population_and_spatial_indices=True, retain_terminal_events_and_voltage_separately=True,
            descriptive_source_analysis_window_ms=[500, 3000], source_analysis_lower_endpoint="open",
            actual_control_transition_from="delivered_movie_and_controls", inferential_statistics=False,
            source_qualification=False, scientific_tolerances=None))
    base = dict(unit_coverage="all_source_units", stimulus=dict(direction_deg=315, duration_s=3.,
        gray_screen_s=.5, contrast_percent=80., spatial_frequency_cycles_per_degree=.04, temporal_frequency_hz=2.),
        spike_generation=dict(trials=1, starting_noise_seed=12345), background=dict(starting_noise_seed=12346))
    protocol_path = configs / "direction.yaml"; protocol_path.write_text(yaml.safe_dump(protocol))
    (configs / "visual.yaml").write_text(yaml.safe_dump(base))
    submission = dict(protocol_sha256=sha256_file(protocol_path), maximum_new_full_conditions=7,
        condition_concurrency=1, source_qualification=False, production_episodes=0,
        jobs=[dict(phase="full_native_and_replay", direction_deg=value, job_id=100 + value,
                   status="SUBMITTED", returncode=0) for value in response.DIRECTIONS[:-1]])
    submission_path = tmp_path / "submission.json"; submission_path.write_text(json.dumps(submission))
    state = dict(source_id=np.array([9, 1, 5]), inverse_source_id=np.array([9, 1, 5]),
        native_id=np.array([19, 11, 15]), node_type_id=np.array([200, 100, 200]),
        initial_vm_mV=np.array([-65., -60., -65.]), terminal_vm_mV=np.array([-64., -59., -66.]),
        spike_count=np.array([2, 2, 0]), population=np.array(["v1"] * 3))
    spikes = dict(source_id=np.array([1, 1, 9, 9]), native_id=np.array([11, 11, 19, 19]),
                  time_ms=np.array([.25, 500., 500.25, 3000.]))
    incoming = dict(source_id=np.array([0, 1, 0]), inverse_source_id=np.array([0, 1, 0]),
        native_id=np.array([20, 21, 22]), source_event_counts=np.array([1, 0, 1]),
        precise_times=np.ones(3, dtype=bool), population=np.array(["lgn", "lgn", "bkg"]),
        source_event_unit=np.array([0, 2]), requested_times_ms=np.array([.25, 3000.]),
        native_times_ms=np.array([.25, 3000.]), encoded_steps=np.array([1, 12000]), encoded_offsets_ms=np.zeros(2))
    input_identity = input_plan_identity(dict(population=np.array([0, 0, 1]), source_id=incoming["source_id"],
        source_counts=incoming["source_event_counts"], source_event_unit=incoming["source_event_unit"],
        source_event_time_ms=incoming["requested_times_ms"]))
    records = {name: dict(bytes=1, sha256=name) for name in response.INPUT_CONSTRUCTION_FILES}
    paths, admissions = {}, {}
    for direction in response.DIRECTIONS:
        path = tmp_path / ("archive_" + str(direction)); native = path / "primary/native"; native.mkdir(parents=True)
        np.savez(native / "cortical_state.npz", **state); np.savez(native / "cortical_spikes.npz", **spikes)
        np.savez(native / "input_events.npz", **incoming)
        (native / "parameters.json").write_text(json.dumps({"100": dict(all_units_exact=True, units=1, parameters=dict(V_m=-60.)),
                                                           "200": dict(all_units_exact=True, units=2, parameters=dict(V_m=-65.))}))
        types = path / response.SOURCE_TYPE_CSV; types.parent.mkdir(parents=True)
        types.write_text("node_type_id pop_name population\n100 excitatory v1\n200 inhibitory v1\n" +
                         "".join(f"{300 + index} empty{index} v1\n" for index in range(19)))
        config = deepcopy(base); config["stimulus"]["direction_deg"] = direction
        report = dict(mode="full", cortical_units=3, simulated_ms=3000., input_mode="constructed_visual_inputs",
            input_event_schema="packed_source_events_v1", source_scientific_fidelity_qualified=False,
            scientific_qualification=False, production_episodes=0, cortical_spike_events=4, silent_cortical_units=1,
            realized_glif_types=2, original_config_sha256="original", source_denominators=dict(v1=dict(edge_rows=10)),
            kernel_rng_seed=1, nest_version="test", local_num_threads=1,
            connections=dict(source_batch_sha256="coupled", physical_synapses=100),
            native_total_connections_including_output_devices=20, wheel=dict(sha256="wheel", bytes=1, version="test"),
            input_readback=dict(generators=3, events=2, silent_generators=1, tics_per_ms=1000.,
                                all_generator_events_exact=True, all_population_and_source_ids_exact=True),
            hostname="software_fixture", setup_seconds=0., simulation_and_output_seconds=0., elapsed_seconds=0.,
            bindings=dict(inputs=dict(delivery_sha256="inputs", files=deepcopy(records)), parameters=dict(identity="parameters"),
                sources=dict(identity="source"), point_config=dict(source_clock=dict(duration_ms=3000., resolution_ms=.25)),
                recurrent=dict(files={"primary/glif/v1_nodes.h5": dict(bytes=1, sha256="graph"),
                    "primary/glif/v1_node_types.csv": dict(bytes=types.stat().st_size, sha256=sha256_file(types))}),
                spike_inputs=dict(config=config, complete_lgn_population=True, input_identity_sha256=input_identity,
                    bundle_status="CONSTRUCTED_VISUAL_INPUTS_COMPLETE_DEVELOPMENT",
                    source_clock=dict(duration_ms=3000., resolution_ms=.25),
                    input_construction_binding=dict(mode="full", delivery_sha256="inputs", files=deepcopy(records)))))
        job = "900" if direction == 315 else str(100 + direction)
        admissions[str(path)] = dict(archive=str(path), report=report, delivery=dict(job_id=job),
                                    consumed={"delivery.json": dict(sha256="delivery-" + job)})
        paths[str(direction)] = path
    def admit(path, *, fine):
        assert fine is False
        value = deepcopy(admissions[str(path)])
        value["replay_report"] = deepcopy(value["report"])
        return value
    monkeypatch.setattr(response, "admit_archive", admit)
    observation = tmp_path / "observation"; (observation / "export/dataset").mkdir(parents=True)
    catalog = dict(binding=dict(source_bindings=[dict(job_id="900", delivery_sha256="delivery-900")]),
        circuits={"circuit": dict(path="circuits/circuit.h5", sha256="static-map", units=3,
            provenance=dict(unit_metadata_identity="graph", coordinate_unit_evidence="coordinate-authority",
                node_types_identity=sha256_file(paths["315"] / response.SOURCE_TYPE_CSV)))})
    catalog_path = observation / "export/dataset/dataset.json"; catalog_path.write_text(json.dumps(catalog))
    exported = dict(status="EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION", exit_code=0, source_execution_mode="full",
        native_job_id="900", circuit_id="circuit", source_units=3, coordinate_units="um", dataset_catalog_sha256=sha256_file(catalog_path))
    (observation / "export/report.json").write_text(json.dumps(exported))
    (observation / "delivery.json").write_text(json.dumps(dict(exit_code=0, files=[dict(path=name,
        bytes=(observation / name).stat().st_size, sha256=sha256_file(observation / name))
        for name in response.SPATIAL_REFERENCE_FILES[1:]])))
    return dict(protocol=protocol_path, submission=submission_path, paths=paths, admissions=admissions,
                output=tmp_path / "readout", state=state, spikes=spikes, incoming=incoming)


def run(campaign, **kwargs):
    return response.summarize(campaign["protocol"], campaign["submission"], campaign["paths"], campaign["output"], **kwargs)


def test_all_eight_conditions_retain_silent_units_and_exact_window_boundaries(campaign):
    assert set(campaign["spikes"]) == {"source_id", "native_id", "time_ms"}
    report = run(campaign)
    assert report["complete_eight_direction_coverage"] and report["admitted_conditions"] == 8
    assert not report["scientific_qualification"] and not report["production_eligible"]
    assert not report["inferential_statistics"] and report["production_episodes"] == 0
    with np.load(campaign["output"] / "per_unit.npz") as arrays:
        assert arrays["direction_deg"].tolist() == list(response.DIRECTIONS)
        assert arrays["source_id"].tolist() == [1, 5, 9]
        assert arrays["gray_spike_count"].tolist() == [[2, 0, 0]] * 8
        assert arrays["grating_spike_count"].tolist() == [[0, 0, 2]] * 8
        assert arrays["terminal_event_count"].tolist() == [[0, 0, 1]] * 8
        assert arrays["gray_rate_hz"][0].tolist() == [4., 0., 0.]
        assert arrays["grating_rate_hz"][0].tolist() == [0., 0., .8]
        assert arrays["terminal_vm_mV"][0].tolist() == [-59., -66., -64.]
        assert np.array_equal(arrays["full_spike_count"], arrays["gray_spike_count"] + arrays["grating_spike_count"])
    assert all(row["units"] == 3 and row["full_silent_units"] == 1 and row["terminal_spikes"] == 1 for row in report["all_units"])
    assert report["spatial_mapping"]["circuit_record"]["sha256"] == "static-map"


def test_intake_cannot_claim_eight_direction_completion(campaign):
    campaign["paths"] = {"315": campaign["paths"]["315"]}
    report = run(campaign, intake_only=True)
    assert report["status"] == "V1_DIRECTION_INTAKE_ONLY"
    assert report["admitted_conditions"] == 1 and not report["complete_eight_direction_coverage"]


def test_native_clock_annotations_preserve_old_reference_compatibility(campaign):
    current = campaign["admissions"][str(campaign["paths"]["0"])]["report"]["connections"]
    current.update(source_batch_resolution_ms=.25, reference_clock_resolution_ms=.25,
                   reference_clock_source_batch_sha256=current["source_batch_sha256"])
    assert run(campaign)["complete_eight_direction_coverage"]


@pytest.mark.parametrize("field", ["source_batch_resolution_ms", "reference_clock_resolution_ms",
                                   "reference_clock_source_batch_sha256"])
def test_incorrect_connection_clock_annotations_are_rejected(campaign, field):
    current = campaign["admissions"][str(campaign["paths"]["0"])]["report"]["connections"]
    current[field] = "other_batch" if field.endswith("sha256") else .125
    with pytest.raises(ValueError, match="REQUIRED_CONDITIONS_FAILED:0"):
        run(campaign)
    failed = json.loads((campaign["output"] / "report.json").read_text())
    assert failed["admitted_conditions"] == 7
    assert "SOURCE_CLOCK_CONNECTION_ANNOTATION_DIFFERS" in failed["failures"][0]["error"]
    assert not (campaign["output"] / "per_unit.npz").exists()


@pytest.mark.parametrize("defect", ["missing", "duplicate_path", "duplicate_key", "extra"])
def test_archive_coverage_is_exact(campaign, defect):
    if defect == "missing": del campaign["paths"]["0"]
    if defect == "duplicate_path": campaign["paths"]["0"] = campaign["paths"]["315"]
    if defect == "duplicate_key": campaign["paths"][0] = campaign["paths"]["0"]
    if defect == "extra": campaign["paths"]["360"] = Path("unused")
    with pytest.raises(ValueError, match="DIRECTION_.*ARCHIVE"):
        run(campaign)
    assert not (campaign["output"] / "per_unit.npz").exists()


@pytest.mark.parametrize("defect", ["duplicate_direction", "duplicate_job", "wrong_job", "protocol"])
def test_authoritative_submission_binds_conditions_and_jobs(campaign, defect):
    submission = json.loads(campaign["submission"].read_text())
    if defect == "duplicate_direction": submission["jobs"][1]["direction_deg"] = 0
    if defect == "duplicate_job": submission["jobs"][1]["job_id"] = 100
    if defect == "wrong_job": submission["jobs"][0]["job_id"] = 999
    if defect == "protocol": submission["protocol_sha256"] = "changed"
    campaign["submission"].write_text(json.dumps(submission))
    with pytest.raises(ValueError): run(campaign)
    failed = json.loads((campaign["output"] / "report.json").read_text())
    assert failed["exit_code"] == 1 and not failed["complete_eight_direction_coverage"]


@pytest.mark.parametrize("defect", ["direction", "contrast", "noise", "construction", "graph", "coupling", "wheel", "clock", "canary", "denominator", "timing"])
def test_changed_condition_or_native_context_is_rejected(campaign, defect):
    report = campaign["admissions"][str(campaign["paths"]["0"])]["report"]
    binding = report["bindings"]
    if defect == "direction": binding["spike_inputs"]["config"]["stimulus"]["direction_deg"] = 315
    if defect == "contrast": binding["spike_inputs"]["config"]["stimulus"]["contrast_percent"] = 50
    if defect == "noise": binding["spike_inputs"]["config"]["background"]["starting_noise_seed"] += 1
    if defect == "construction": binding["spike_inputs"]["input_construction_binding"]["delivery_sha256"] = "other"
    if defect == "graph": binding["recurrent"]["files"]["primary/glif/v1_nodes.h5"]["sha256"] = "other"
    if defect == "coupling": report["connections"]["source_batch_sha256"] = "other"
    if defect == "wheel": report["wheel"]["sha256"] = "other"
    if defect == "clock": binding["point_config"]["source_clock"]["resolution_ms"] = .125
    if defect == "canary": report["mode"] = "canary"
    if defect == "denominator": report["cortical_units"] = 2
    if defect == "timing": del report["setup_seconds"]
    with pytest.raises(ValueError, match="REQUIRED_CONDITIONS_FAILED:0"): run(campaign)
    failed = json.loads((campaign["output"] / "report.json").read_text())
    assert [row["direction_deg"] for row in failed["failures"]] == [0]
    assert failed["admitted_conditions"] == 7 and not (campaign["output"] / "per_unit.npz").exists()


def test_rejected_reference_cannot_admit_remaining_directions(campaign):
    del campaign["admissions"][str(campaign["paths"]["315"])]["report"]["setup_seconds"]
    with pytest.raises(ValueError, match="REQUIRED_CONDITIONS_FAILED"):
        run(campaign)
    failed = json.loads((campaign["output"] / "report.json").read_text())
    assert failed["admitted_conditions"] == 0
    assert len(failed["failures"]) == 8
    assert not (campaign["output"] / "per_unit.npz").exists()


@pytest.mark.parametrize("defect", ["unknown_id", "native_id", "event_id_dtype", "state_population", "duplicate_id", "counts", "off_clock", "initial_state", "input_identity"])
def test_invalid_events_identities_and_initial_states_cannot_be_hidden(campaign, defect):
    native = campaign["paths"]["0"] / "primary/native"
    state, spikes, incoming = [deepcopy(campaign[key]) for key in ("state", "spikes", "incoming")]
    if defect == "unknown_id": spikes["source_id"][0] = 999
    if defect == "native_id": spikes["native_id"][0] = 999
    if defect == "event_id_dtype": spikes["source_id"] = spikes["source_id"].astype(np.float64)
    if defect == "state_population": state["population"][0] = "xx"
    if defect == "duplicate_id": state["source_id"][0] = 1
    if defect == "counts": state["spike_count"][0] += 1
    if defect == "off_clock": spikes["time_ms"][0] = .125
    if defect == "initial_state": state["initial_vm_mV"][0] += 1
    if defect == "input_identity":
        incoming["requested_times_ms"][0] = .5; incoming["native_times_ms"][0] = .5; incoming["encoded_steps"][0] = 2
    np.savez(native / "cortical_state.npz", **state); np.savez(native / "cortical_spikes.npz", **spikes)
    np.savez(native / "input_events.npz", **incoming)
    with pytest.raises(ValueError, match="REQUIRED_CONDITIONS_FAILED:0"): run(campaign)


def test_json_duplicate_condition_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "archives.json"; path.write_text('{"0":"first","0":"second"}')
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        response._json(path)
