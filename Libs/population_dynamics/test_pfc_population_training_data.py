"""Actual optimizer/resume checks with disposable, distinct source maps."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from . import population_trajectory_training as training
from . import renewal_trajectory_training as shared
from . import pfc_population_training_data as consumer
from .population_state import ControlledPopulationState
from .test_population_metadata import model
from . import test_pfc_source_dataset as source_fixtures
from . import pfc_source_dataset as exporter
from . import pfc_scale_analysis as source_analysis


def configuration():
    return dict(schema="pfc_population_development_training_v1", model_family="population",
        training_seeds=[41, 42, 43], input_family="pfc_source_reproduction",
        training_sequence="complete_source_episode", observation_objective="native_event_time_mc_negative_elbo",
        partition_scope="whole_circuit_development_only", formal_training_completed=False,
        validation_randomness="fixed_per_training_seed_case_and_draw",
        variational_rng="explicit_separate_checkpointed_generator", fit_circuit_ids=["fixture_7_3", "fixture_11_4"],
        validation_circuit_ids=["fixture_9_2"], latent_channels=2, encoder_size=5, unit_chunk_size=3,
        bin_ticks=5, maximum_epochs=4, early_stopping_patience=4, validation_draws=2,
        cpu_threads=1, representation="recurrent", learning_rate=.001,
        initialization_pseudocount=.5, gradient_norm_limit=100., device="cpu",
        training_variational_seed_base=61001, validation_seed_base=64001)


class DifferentMapCohort(consumer.PfcPopulationTrainingCohort):
    """Replace only source admission; retain the production numerical consumer."""
    validation_multiplier = 1

    def __init__(self, path, config):
        self.config = config
        self.fit = [dict(case_id="fit_seven", circuit_id="fixture_7_3", n=7, f=3),
                    dict(case_id="fit_eleven", circuit_id="fixture_11_4", n=11, f=4)]
        self.validation = [dict(case_id="validate_nine", circuit_id="fixture_9_2", n=9, f=2)]
        self.unit_ids = np.arange(7)
        self.binding = dict(fixture=True, partitions=[config["fit_circuit_ids"], config["validation_circuit_ids"]])

    def population_configuration(self, config, seed):
        value = model().configuration()
        value["initialization_seed"] = seed
        return value

    def window(self, case):
        self.active_case = case
        counts = np.zeros((8, case["n"]), dtype=np.int64)
        counts[1::3, ::2] = self.validation_multiplier if case in self.validation else 1
        return SimpleNamespace(counts=counts, eligible_ticks=np.full_like(counts, 5),
                               stimulus=np.zeros((8, 1)))

    def model_for_case(self, canonical, case):
        assert self.active_case == case
        other = model(case["n"], case["f"])
        return canonical.training_for_circuit(unit_ids=other.unit_ids,
            unit_field_index=other.unit_field_index.tolist(), field_names=other.field_names,
            minimum_interval_ticks=other.minimum_interval_ticks.tolist(), metadata=other.metadata)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): same(a, b)
    else:
        assert left == right


def checkpoint(path):
    return torch.load(path / "latest.pt", weights_only=True)


@pytest.fixture
def numerical_fixture(tmp_path, monkeypatch):
    config = configuration()
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(consumer, "PfcPopulationTrainingCohort", DifferentMapCohort)
    return config, path


def test_actual_multicircuit_trainer_resume_has_identical_optimizer_rng_and_losses(tmp_path, numerical_fixture):
    _, config = numerical_fixture
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    training.train(tmp_path, None, config, continuous, 42, stop_after_epochs=3)
    training.train(tmp_path, None, config, split, 42, stop_after_epochs=1)
    report = training.train(tmp_path, None, config, split, 42, resume=True, stop_after_epochs=2)
    left, right = checkpoint(continuous), checkpoint(split)
    for key in ("model", "optimizer", "variational_rng", "history", "next_epoch", "best", "best_epoch", "initialization"):
        same(left[key], right[key])
    assert len(left["optimizer"]["state"]) == len(list(ControlledPopulationState(**left["model_configuration"]).parameters()))
    assert all(int(value["step"]) == 6 for value in left["optimizer"]["state"].values())
    assert report["fitting_circuits"] == 2 and report["validation_circuits"] == 1
    assert not report["formal_training_completed"] and not report["test_outcomes_accessed"]


def test_fit_only_shared_hazard_matches_direct_weighted_least_squares(numerical_fixture):
    config, _ = numerical_fixture
    cohort = DifferentMapCohort(None, config)
    canonical = ControlledPopulationState(**cohort.population_configuration(config, 41))
    matrix, target = [], []
    for case in cohort.fit:
        window = cohort.window(case)
        view = cohort.model_for_case(canonical, case)
        successes, eligible = window.counts.sum(0), window.eligible_ticks.sum(0)
        probability = (successes + .5) / (eligible + 1)
        weights = np.sqrt(eligible + 1)
        matrix.append(np.column_stack((view.unit_features.numpy(), np.ones(case["n"]))) * weights[:, None])
        target.append(np.log(-np.log1p(-probability) / .001) * weights)
    expected = np.linalg.lstsq(np.concatenate(matrix), np.concatenate(target), rcond=None)[0]
    report = cohort.initialize_observation(canonical, .5)
    np.testing.assert_allclose(report["coefficients"], expected, rtol=1e-12, atol=1e-12)
    second = DifferentMapCohort(None, config); second.validation_multiplier = 4
    another = ControlledPopulationState(**second.population_configuration(config, 41))
    same(report, second.initialize_observation(another, .5))
    assert report["fitted_source_units"] == 18 and not report["validation_observations_used"]


def test_streamed_hazard_rank_matches_full_design_with_constant_source_descriptors(numerical_fixture):
    config, _ = numerical_fixture
    class ConstantDescriptorCohort(DifferentMapCohort):
        def model_for_case(self, canonical, case):
            assert self.active_case == case
            units = case["n"]
            features = np.column_stack((np.arange(units) / units,
                np.ones(units), np.full(units, 12.5), np.full(units, 2.3))).astype(np.float32)
            values = torch.as_tensor(features)
            return SimpleNamespace(unit_features=values, unit_ids=tuple(range(len(values))),
                                   _intrinsic_unit_features=lambda start, stop: values[start:stop])
    cohort = ConstantDescriptorCohort(None, config)
    cohort.fit = [dict(case_id="large_fit", circuit_id="fixture_large", n=2000, f=3)]
    canonical = SimpleNamespace(unit_features=torch.zeros(2000, 4), tick_seconds=.001,
        unit_chunk_size=3, shared_unit_readout=torch.nn.Linear(4, 3))
    window = cohort.window(cohort.fit[0])
    features = cohort.model_for_case(canonical, cohort.fit[0]).unit_features.numpy().astype(np.float64)
    weights = np.sqrt(window.eligible_ticks.sum(0) + 1)
    design = np.column_stack((features, np.ones(2000))) * weights[:, None]
    probability = (window.counts.sum(0) + .5) / (window.eligible_ticks.sum(0) + 1)
    response = np.log(-np.log1p(-probability) / .001) * weights
    expected, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    report = cohort.initialize_observation(canonical, .5)
    assert report["descriptor_rank"] == rank == 2
    np.testing.assert_allclose(report["coefficients"], expected, rtol=1e-9, atol=1e-9)
    log_hazard = canonical.shared_unit_readout(torch.as_tensor(features,dtype=torch.float32))[:, -1]
    assert torch.isfinite(log_hazard).all() and log_hazard.abs().max() < 10


def test_validation_uses_its_map_and_separate_fixed_rng(numerical_fixture):
    config, _ = numerical_fixture
    cohort = DifferentMapCohort(None, config)
    canonical = ControlledPopulationState(**cohort.population_configuration(config, 41))
    rng = torch.get_rng_state().clone()
    first = shared.evaluate_validation(canonical, cohort, config, 41)
    assert first == shared.evaluate_validation(canonical, cohort, config, 41)
    assert torch.equal(rng, torch.get_rng_state())


def test_hazard_initialization_retains_distinct_target_geometry_groups(numerical_fixture):
    config, _ = numerical_fixture
    class GeometryCohort(DifferentMapCohort):
        def model_for_case(self, canonical, case):
            features = np.column_stack((np.arange(case["n"]) / case["n"],
                np.isin(np.arange(case["n"]), case["source_target_geometry"]))).astype(np.float32)
            values = torch.as_tensor(features)
            return SimpleNamespace(unit_features=values, unit_ids=tuple(range(len(values))),
                                   _intrinsic_unit_features=lambda start, stop: values[start:stop])
    cohort = GeometryCohort(None, config)
    cohort.fit = [dict(case_id="left", circuit_id="one_circuit", n=7, source_target_geometry=(0,1)),
                  dict(case_id="right", circuit_id="one_circuit", n=7, source_target_geometry=(4,5))]
    canonical = SimpleNamespace(unit_features=torch.zeros(7,2), tick_seconds=.001,
        unit_chunk_size=3, shared_unit_readout=torch.nn.Linear(2,3))
    design, response = [], []
    for case in cohort.fit:
        window = cohort.window(case)
        features = cohort.model_for_case(canonical,case).unit_features.numpy().astype(np.float64)
        eligible, successes = window.eligible_ticks.sum(0), window.counts.sum(0)
        weight = np.sqrt(eligible + 1)
        design.append(np.column_stack((features,np.ones(7))) * weight[:,None])
        response.append(np.log(-np.log1p(-(successes+.5)/(eligible+1))/.001)*weight)
    expected = np.linalg.lstsq(np.concatenate(design),np.concatenate(response),rcond=None)[0]
    result = cohort.initialize_observation(canonical,.5)
    np.testing.assert_allclose(result["coefficients"],expected,rtol=1e-12,atol=1e-12)
    assert result["fitting_source_geometry_groups"] == 2
    assert result["fitted_source_units"] == 7 and result["fitted_source_feature_rows"] == 14


@pytest.mark.parametrize("intended_split",["calibration","test"])
def test_prospective_protected_circuit_rejects_before_source_admission(tmp_path,monkeypatch,intended_split):
    class Catalog:
        def __init__(self,*args,**kwargs):
            self.catalog = {"binding":dict(scope="source_reproduction",scenario_id="pfc_working_memory",
                resolved_source_config={"benchmark_episode_candidate":{"intended_split":intended_split}})}
            self.episodes = {"protected":dict(circuit_id="fixture_7_3")}
    source = tmp_path/"source.py"; source.write_text("source")
    spec = tmp_path/"inputs.json"
    spec.write_text(json.dumps(dict(schema="pfc_population_development_inputs_v1",source_entry=str(source),
        episodes=[dict(dataset="fixture",episode_id="protected",source_run="never_read",partition="development_fit")])))
    monkeypatch.setattr(consumer,"CircuitDataset",Catalog)
    monkeypatch.setattr(consumer,"sha256_file",lambda path:consumer.SOURCE_ENTRY_SHA256)
    monkeypatch.setattr(consumer,"PfcRenewalEpisode",lambda *args,**kwargs:pytest.fail("protected source read"))
    with pytest.raises(ValueError,match="PROSPECTIVE_CIRCUIT_SPLIT_CHANGED"):
        consumer.PfcPopulationTrainingCohort(spec,configuration())


def test_partition_and_resume_changes_reject_before_output_write(tmp_path, numerical_fixture):
    config, path = numerical_fixture
    output = tmp_path / "fit"
    training.train(tmp_path, None, path, output, 41, stop_after_epochs=1)
    original = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    config["validation_circuit_ids"] = ["fixture_7_3"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="SPLIT_LEAKAGE"):
        training.train(tmp_path, None, path, output, 41, resume=True)
    config["validation_circuit_ids"] = ["fixture_10_2"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="binding changed"):
        training.train(tmp_path, None, path, output, 41, resume=True)
    assert original == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_resume_rejects_modified_source_buffer_even_with_updated_receipt(tmp_path, numerical_fixture):
    _, config = numerical_fixture
    output = tmp_path / "fit"
    training.train(tmp_path, None, config, output, 43, stop_after_epochs=1)
    payload = checkpoint(output)
    payload["model"]["unit_features"][0, 0] += 1
    torch.save(payload, output / "latest.pt")
    from .provenance import sha256_file
    report = json.loads((output / "run.json").read_text())
    report["latest_checkpoint_sha256"] = sha256_file(output / "latest.pt")
    (output / "run.json").write_text(json.dumps(report))
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="source buffers changed"):
        training.train(tmp_path, None, config, output, 43, resume=True)
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_real_source_admission_requires_complete_declared_circuit_partition(tmp_path):
    fixture = source_fixtures.PfcSourceDatasetTests(); fixture.setUp()
    try:
        source = fixture.source()
        exporter.export([source], fixture.config, fixture.root / "export")
        dataset = exporter.CircuitDataset(fixture.root / "export/dataset", splits=("development",))
        circuit = next(dataset.iter_episodes())["circuit_id"]
        config = configuration(); config["fit_circuit_ids"] = [circuit]
        spec = dict(schema="pfc_population_development_inputs_v1",
            source_entry=str(Path(__file__).resolve().parents[1] / "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py"),
            episodes=[dict(dataset=str(dataset.root), episode_id="pfc_94001_strong_cue", source_run=str(source),
                           partition="development_fit")])
        path = tmp_path / "inputs.json"; path.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="DENOMINATOR_INCOMPLETE"):
            consumer.PfcPopulationTrainingCohort(path, config)
        spec["episodes"] *= 2; path.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="DUPLICATE_COMPLETE_EPISODE"):
            consumer.PfcPopulationTrainingCohort(path, config)
    finally:
        fixture.doCleanups()


def test_observed_source_domain_registration_preserves_frozen_numerical_protocol(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/neural_population_pfc_recurrent_gain_expanded_100k.yaml"
    config, registered, seeds, dimensions = source_analysis.observed_source_registration(path)
    assert len(registered) == 9 and seeds == [95001, 95002, 95003]
    assert dimensions["coupled_core_units"] == 100000 and config["Jpos_excit2excit"] == 2.3
    with pytest.raises(ValueError, match="numerical calibration replication"):
        source_analysis._source_registration(path)
    changed = tmp_path / "modified.yaml"
    changed.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="FROZEN_GAIN_SOURCE_CONFIG"):
        source_analysis.observed_source_registration(changed)


def test_source_bound_cli_fits_complete_distinct_circuits_and_samples_after_resume(tmp_path):
    fixtures, records, circuits = [], [], []
    try:
        for index, case in enumerate(("pfc_94001_strong_cue", "pfc_94002_strong_cue")):
            fixture = source_fixtures.PfcSourceDatasetTests(); fixture.setUp(); fixtures.append(fixture)
            if index:
                fixture.fixture.config.update(sigma_weight_profile=30., Jpos_excit2excit=1.7)
                fixture.fixture.config["excitatory_preference_bins"] = 2
                fixture.config.write_text(yaml.safe_dump(fixture.fixture.config))
            source = fixture.source(case)
            if index: fixture.fixture.parameterize_archive(source)
            exporter.export([source], fixture.config, fixture.root / "export")
            dataset = exporter.CircuitDataset(fixture.root / "export/dataset", splits=("development",))
            circuits.append(next(dataset.iter_episodes())["circuit_id"])
            records.append(dict(dataset=str(dataset.root), episode_id=case, source_run=str(source),
                partition="development_fit" if not index else "development_validation"))
        config = configuration()
        config.update(fit_circuit_ids=circuits[:1], validation_circuit_ids=circuits[1:],
                      bin_ticks=100, maximum_epochs=2, validation_draws=1)
        path = tmp_path / "training.yaml"; path.write_text(yaml.safe_dump(config))
        inputs = tmp_path / "inputs.json"
        inputs.write_text(json.dumps(dict(schema="pfc_population_development_inputs_v1", episodes=records,
            source_entry=str(Path(__file__).resolve().parents[1] / "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py"))))
        # Fixture-only source code admission is scoped to this child process.
        launcher = '''import hashlib,sys
from population_dynamics import pfc_scale_analysis, population_trajectory_training
pfc_scale_analysis._LEGACY_DEFAULT_KERNEL_EXECUTORS = pfc_scale_analysis._LEGACY_DEFAULT_KERNEL_EXECUTORS | {hashlib.sha256(b"# Engineering fixture, not executable source: pfc_deterministic_replay.py\\n").hexdigest()}
population_trajectory_training.main()
'''
        output = tmp_path / "training"
        command = [sys.executable, "-c", launcher, "--acquisition", str(inputs), "--config", str(path),
                   "--output", str(output), "--seed", "41", "--stop-after-epochs", "1"]
        first = subprocess.run(command, text=True, capture_output=True)
        assert first.returncode == 0, first.stderr
        first_payload = checkpoint(output)
        second = subprocess.run(command + ["--resume"], text=True, capture_output=True)
        assert second.returncode == 0, second.stderr
        payload = checkpoint(output)
        assert payload["next_epoch"] == 2
        assert payload["initialization"] == first_payload["initialization"]
        canonical = ControlledPopulationState(**payload["model_configuration"])
        canonical.load_state_dict(payload["model"])
        cohort = consumer.PfcPopulationTrainingCohort(inputs, config)
        case = cohort.validation[0]; window = cohort.window(case)
        view = cohort.model_for_case(canonical, case)
        assert len(view.field_names) == 3 and len(canonical.field_names) == 5
        history = dict(counts=torch.as_tensor(window.counts[:2])[None],
                       stimulus=torch.as_tensor(window.stimulus[:2])[None],
                       ready_in_ticks=torch.zeros(1, 20, dtype=torch.long))
        boundary = view.infer_state(history)
        from .poisson_state_space import LatentRandomStreams
        def generate():
            return next(view.iter_simulation(boundary, torch.as_tensor(window.stimulus[2:3])[None],
                        LatentRandomStreams.from_seeds(77001, 77002, 77003), 1))
        a, b = generate(), generate()
        assert torch.equal(a.counts, b.counts) and torch.equal(a.events, b.events)
        report = json.loads((output / "run.json").read_text())
        assert report["input_family"] == "pfc_source_reproduction" and report["resumed"]
        assert not report["formal_training_completed"] and not report["generative_fidelity_qualified"]
    finally:
        for fixture in fixtures: fixture.doCleanups()
