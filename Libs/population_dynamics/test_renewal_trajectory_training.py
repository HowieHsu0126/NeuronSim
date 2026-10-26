"""CPU contract tests; tiny inputs replace only the production cohort boundary."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from . import renewal_trajectory_training as training
from .native_renewal_data import NATIVE_SOURCE_REVISION, NativeRenewalEpisode, load_dataset_index
from .provenance import sha256_file
from .refractory_observation import renewal_window
from .renewal_state_space import ControlledRenewalLDS
from .test_poisson_trajectory_training import make_legacy_parent, snapshot


def _json(path, value):
    path.write_text(json.dumps(value))


def _native_fixture(root, *, legacy=False):
    episode, fields = root / "native", root / "fields"
    episode.mkdir(parents=True)
    fields.mkdir()
    np.savez(episode / "spikes.npz", unit_ids=np.array([3, 7, 12, 20]),
             populations=np.array(["E", "I", "E", "TC"]),
             spike_ticks=np.array([30, 50, 55, 100, 200]),
             spike_unit_ids=np.array([3, 7, 3, 20, 12]))
    np.savez(episode / "initial_parameters.npz", unit_ids=np.array([3, 7, 12]),
             V_reset_rel=np.zeros(3), Theta_rel=np.full(3, 20.), t_ref=np.full(3, 2.))
    report = dict(status="EXECUTED", gpu_source_revision=NATIVE_SOURCE_REVISION,
                  native_library_sha256="a" * 64, graph_manifest_sha256="b" * 64,
                  dt_ms=.1, duration_ms=20.,
                  artifacts={name: sha256_file(episode / name) for name in ("spikes.npz", "initial_parameters.npz")})
    if not legacy:
        report["native_poisson_rng_family"] = "per_edge_philox4_32_10"
    _json(episode / "report.json", report)
    np.savez(fields / "spatial_fields.npz", source_unit_ids=np.array([3, 7, 12]),
             unit_field_index=np.array([0, 1, 0]), field_ids=np.array(["E_0", "I_0"]))
    _json(fields / "summary.json", dict(status="DESCRIBED", settings=dict(bin_ms=5.), dt_ms=.1,
          window_start_tick=0, window_stop_tick=200,
          episode_report_sha256=sha256_file(episode / "report.json"),
          artifacts={"spatial_fields.npz": sha256_file(fields / "spatial_fields.npz")}))
    return episode, fields


def test_native_boundary_is_strictly_causal_and_exposure_exact(tmp_path):
    data = NativeRenewalEpisode(*_native_fixture(tmp_path))
    np.testing.assert_array_equal(data.minimum_interval_ticks, [21, 21, 21])
    np.testing.assert_array_equal(data.ready_at(50), [1, 0, 0])
    np.testing.assert_array_equal(data.ready_at(51), [0, 20, 0])
    window = data.renewal_window(50, 100)
    np.testing.assert_array_equal(window.counts, [[1, 1, 0]])
    np.testing.assert_array_equal(window.eligible_ticks, [[29, 30, 50]])
    np.testing.assert_array_equal(window.initial_ready_in_ticks, data.ready_at(50))
    assert data.renewal_window(0, 200).counts[:, 2].sum() == 0
    # A later invalid event cannot contaminate the earlier availability.
    data.event_ticks = np.append(data.event_ticks, 201)
    data.event_units = np.append(data.event_units, 2)
    np.testing.assert_array_equal(data.ready_at(50), [1, 0, 0])


def test_native_identity_map_and_source_changes_rejected(tmp_path):
    paths = _native_fixture(tmp_path)
    data = NativeRenewalEpisode(*paths)
    for keyword, bad in [("expected_unit_ids", [7, 3, 12]),
                         ("expected_unit_field_index", [1, 0, 0]),
                         ("expected_source", dict(data.source_identity, native_library_sha256="c" * 64))]:
        with pytest.raises(ValueError):
            NativeRenewalEpisode(*paths, **{keyword: bad})
    with (paths[0] / "initial_parameters.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="initial parameters changed"):
        NativeRenewalEpisode(*paths)


def test_legacy_source_family_remains_explicitly_unreported(tmp_path):
    data = NativeRenewalEpisode(*_native_fixture(tmp_path, legacy=True))
    assert data.source_identity["native_poisson_rng_family"] == "not_reported_by_native_library"


def test_dataset_index_hash_and_duplicate_identity_rejection(tmp_path):
    rows = [dict(case_id=f"case_{i}", split="train", raw_report_sha256="a" * 64,
                 field_summary_sha256="b" * 64) for i in range(2)]
    def write_table():
        with (tmp_path / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _json(tmp_path / "dataset.json", dict(status="BUILT", split="train", completed_episodes=2,
              episode_table_sha256=sha256_file(tmp_path / "episodes.csv")))
    write_table()
    assert set(load_dataset_index(tmp_path)) == {"case_0", "case_1"}
    with (tmp_path / "episodes.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="unchanged episode table"):
        load_dataset_index(tmp_path)
    rows[1] = rows[0].copy()
    write_table()
    with pytest.raises(ValueError, match="unique case identities"):
        load_dataset_index(tmp_path)


class TinyCohort:
    """The CPU fixture does not alter or bypass any production gate in code."""
    marker = "fixture_source_v1"

    def __init__(self, acquisition, fields, config):
        self.fit = [dict(case_id="fit_a"), dict(case_id="fit_b")]
        self.validation = [dict(case_id="validation_a")]
        self.unit_ids = np.array([3, 7, 12])
        self.unit_field_index = np.array([0, 1, 0])
        self.field_ids = np.array(["E_0", "I_0"])
        self.minimum_interval_ticks = np.array([21, 21, 21])
        self.dt_ms, self.bin_ms, self.duration_ticks = .1, 5., 200
        self.binding = dict(acquisition_design_sha256="a" * 64, dataset_sha256="b" * 64,
                            native_source_identity=dict(fixture=self.marker))

    def window(self, case):
        data = object.__new__(NativeRenewalEpisode)
        data.unit_ids, data.bin_ticks, data.duration_ticks = self.unit_ids, 50, self.duration_ticks
        data.pulse_ticks = np.array([50, 150])
        offset = 2 if case["case_id"] == "fit_b" else 0
        data.event_ticks = np.array([10, 30, 50, 75, 130, 180]) + offset
        data.event_units = np.array([0, 1, 0, 2, 1, 0])
        return renewal_window(data, 0, data.duration_ticks, self.minimum_interval_ticks)


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    config = copy.deepcopy(training.FIXED_BUDGET)
    config.update(training_variational_seed_base=61001, validation_seed_base=64001,
                  device="cpu", cpu_threads=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(training, "TrainingCohort", TinyCohort)
    return config, path


def _checkpoint(path):
    return torch.load(path / "latest.pt", map_location="cpu", weights_only=True)


def _same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _restored_loss(checkpoint):
    model = ControlledRenewalLDS(**checkpoint["model_configuration"])
    model.load_state_dict(checkpoint["model"])
    generator = torch.Generator().set_state(checkpoint["variational_rng"])
    cohort = TinyCohort(None, None, None)
    inputs = training._inputs(cohort.window(cohort.fit[0]), torch.device("cpu"))
    return training._numbers(model.sequence_loss(*inputs, generator))


def test_split_resume_exact_optimizer_rng_and_next_loss(tmp_path, training_fixture):
    _, config = training_fixture
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    training.train(tmp_path, tmp_path, config, continuous, 41, stop_after_epochs=3)
    result = training.train(tmp_path, tmp_path, config, split, 41, stop_after_epochs=1)
    assert result["status"] == "CHECKPOINTED_DEVELOPMENT"
    training.train(tmp_path, tmp_path, config, split, 41, resume=True, stop_after_epochs=2)
    a, b = _checkpoint(continuous), _checkpoint(split)
    for key in ("model", "optimizer", "variational_rng", "history", "next_epoch", "best", "stale", "best_epoch"):
        _same(a[key], b[key])
    assert _restored_loss(a) == _restored_loss(b)
    assert b["attempt"] == 1
    assert (split / "attempts" / "attempt_000.json").exists()
    with (split / "validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert [r["variational_seed"] for r in rows[:4]] == [r["variational_seed"] for r in rows[4:8]]


def test_validation_does_not_advance_training_or_global_rng(training_fixture):
    config, _ = training_fixture
    cohort = TinyCohort(None, None, config)
    model = ControlledRenewalLDS(unit_ids=cohort.unit_ids.tolist(), input_size=1, latent_size=16,
                                 encoder_size=32, bin_seconds=.005, tick_seconds=.0001,
                                 minimum_interval_ticks=cohort.minimum_interval_ticks, initialization_seed=41)
    generator = torch.Generator().manual_seed(99)
    state, global_state = generator.get_state().clone(), torch.get_rng_state().clone()
    a = training.evaluate_validation(model, cohort, config, 41)
    b = training.evaluate_validation(model, cohort, config, 41)
    assert a == b
    assert torch.equal(state, generator.get_state())
    assert torch.equal(global_state, torch.get_rng_state())


@pytest.mark.parametrize("changed", ["config", "data", "native"])
def test_resume_changed_binding_rejected_without_output_mutation(tmp_path, training_fixture, monkeypatch, changed):
    values, config = training_fixture
    output = tmp_path / "run"
    training.train(tmp_path, tmp_path, config, output, 41, stop_after_epochs=1)
    before = sha256_file(output / "run.json")
    if changed == "config":
        values["validation_seed_base"] += 1
        config.write_text(yaml.safe_dump(values))
    elif changed == "native":
        monkeypatch.setattr(TinyCohort, "marker", "different_native_source")
    else:
        class DifferentData(TinyCohort):
            def __init__(self, *args):
                super().__init__(*args)
                self.binding["dataset_sha256"] = "d" * 64
        monkeypatch.setattr(training, "TrainingCohort", DifferentData)
    with pytest.raises(ValueError, match="binding changed"):
        training.train(tmp_path, tmp_path, config, output, 41, resume=True)
    assert sha256_file(output / "run.json") == before


def test_partial_epoch_failure_is_preserved_and_replayable(tmp_path, training_fixture, monkeypatch):
    _, config = training_fixture
    baseline, failed = tmp_path / "baseline", tmp_path / "failure"
    training.train(tmp_path, tmp_path, config, baseline, 41, stop_after_epochs=2)
    original = ControlledRenewalLDS.sequence_loss
    calls = 0
    def failure(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected partial epoch failure")
        return original(self, *args)
    monkeypatch.setattr(ControlledRenewalLDS, "sequence_loss", failure)
    with pytest.raises(RuntimeError, match="partial epoch"):
        training.train(tmp_path, tmp_path, config, failed, 41, stop_after_epochs=2)
    assert json.loads((failed / "run.json").read_text())["status"] == "FAILED"
    assert _checkpoint(failed)["next_epoch"] == 0
    monkeypatch.setattr(ControlledRenewalLDS, "sequence_loss", original)
    training.train(tmp_path, tmp_path, config, failed, 41, resume=True, stop_after_epochs=2)
    for key in ("model", "optimizer", "variational_rng", "history"):
        _same(_checkpoint(baseline)[key], _checkpoint(failed)[key])
    assert json.loads((failed / "attempts" / "attempt_000.json").read_text())["status"] == "FAILED"


def test_interrupted_best_copy_recovers_exact_selected_epoch(tmp_path, training_fixture, monkeypatch):
    _, config = training_fixture
    output = tmp_path / "interrupted_copy"
    original = training._save_checkpoint
    def fail_best(payload, path):
        if path.name == "best.pt":
            raise OSError("injected best copy interruption")
        return original(payload, path)
    monkeypatch.setattr(training, "_save_checkpoint", fail_best)
    with pytest.raises(OSError, match="best copy"):
        training.train(tmp_path, tmp_path, config, output, 41, stop_after_epochs=1)
    assert _checkpoint(output)["next_epoch"] == 1
    monkeypatch.setattr(training, "_save_checkpoint", original)
    result = training.train(tmp_path, tmp_path, config, output, 41, resume=True, stop_after_epochs=1)
    assert result["completed_epochs"] == 2
    assert sha256_file(output / "best.pt") == result["best_checkpoint_sha256"]


def test_initial_hazard_uses_only_fitting_eligible_exposure(training_fixture):
    config, _ = training_fixture
    cohort = TinyCohort(None, None, config)
    model = ControlledRenewalLDS(unit_ids=cohort.unit_ids.tolist(), input_size=1, latent_size=16, encoder_size=32,
                                 bin_seconds=.005, tick_seconds=.0001,
                                 minimum_interval_ticks=cohort.minimum_interval_ticks, initialization_seed=41)
    fit_windows = [cohort.window(case) for case in cohort.fit]
    successes = sum(w.counts.sum(0) for w in fit_windows)
    exposures = sum(w.eligible_ticks.sum(0) for w in fit_windows)
    expected = np.log(-np.log1p(-(successes + .5) / (exposures + 1.)) / .0001)
    report = training.initialize_observation(model, cohort, .5)
    np.testing.assert_allclose(model.latent_model.log_rate_bias.detach().numpy(), expected, rtol=1e-6)
    assert report["fitting_episodes"] == 2


def test_production_guard_rejects_incomplete_cohort_and_short_episode(tmp_path):
    cases = [dict(case_id=f"block_{i}", block=i, split="train", family="trajectory",
                  template_id="familiar_0") for i in range(18)]
    _json(tmp_path / "design.json", dict(acquired_split="train", cases=cases))
    _json(tmp_path / "dataset.json", {})
    _json(tmp_path / "acquisition.json", {})
    with pytest.raises(ValueError, match="all 72"):
        training.TrainingCohort(tmp_path, tmp_path, training.FIXED_BUDGET)
    cohort = object.__new__(training.TrainingCohort)
    cohort.minimum_interval_ticks = np.full(100000, 21)
    cohort.field_ids = np.array(["field"])
    short = SimpleNamespace(unit_ids=np.arange(100000), bin_ms=5., dt_ms=.1,
                            duration_ticks=1000, bin_ticks=50,
                            minimum_interval_ticks=cohort.minimum_interval_ticks, field_ids=cohort.field_ids)
    with pytest.raises(ValueError, match="1200-by-100000"):
        cohort._check_episode(short)


def test_frozen_budget_and_training_seed_remain_enforced(training_fixture):
    config, _ = training_fixture
    for key, bad in [("maximum_epochs", 1), ("latent_size", 2), ("training_seeds", [41]),
                     ("training_sequence", "truncated"), ("expected_poisson_rng_family", "per_edge_xorwow")]:
        with pytest.raises(ValueError, match="frozen"):
            training.validate_config(dict(config, **{key: bad}), 41)


def test_cli_help_and_invalid_seed_exit():
    command = [sys.executable, "-m", "Libs.population_dynamics.renewal_trajectory_training"]
    result = subprocess.run(command + ["--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "--resume" in result.stdout and "--stop-after-epochs" in result.stdout
    bad = subprocess.run(command + ["--seed", "44"], text=True, capture_output=True)
    assert bad.returncode != 0


def test_complete_metadata_preflight_and_renamed_case_rejection(tmp_path, monkeypatch):
    acquisition, fields = tmp_path / "acquisition", tmp_path / "fields"
    acquisition.mkdir()
    fields.mkdir()
    cases, rows = [], []
    for block in range(18):
        for template in range(4):
            case = dict(case_id=f"train_block_{block:03d}_familiar_{template}", block=block,
                        template_id=f"familiar_{template}", split="train", family="trajectory", settings=dict(block=block, template=template))
            cases.append(case)
            raw, view = acquisition / case["case_id"], fields / case["case_id"]
            raw.mkdir()
            view.mkdir()
            _json(raw / "report.json", dict(status="EXECUTED", native_library_sha256="a" * 64,
                  native_poisson_rng_family="per_edge_philox4_32_10", graph_manifest_sha256="b" * 64,
                  config=case["settings"], duration_ms=6000., dt_ms=.1))
            _json(view / "summary.json", dict(case_id=case["case_id"]))
            rows.append(dict(case_id=case["case_id"], block=block, split="train", family="trajectory",
                             template_id=case["template_id"], cortical_units=100000,
                             raw_report_sha256=sha256_file(raw / "report.json"),
                             field_summary_sha256=sha256_file(view / "summary.json")))
    _json(acquisition / "design.json", dict(acquired_split="train", cases=cases,
          native_library_sha256="a" * 64, graph_manifest_sha256="b" * 64,
          config=dict(execution=dict(expected_poisson_rng_family="per_edge_philox4_32_10"))))
    _json(acquisition / "acquisition.json", dict(status="ACQUIRED", completed_episodes=72,
          runs=[dict(case_id=c["case_id"], exit_code=0) for c in cases]))
    with (fields / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _json(fields / "dataset.json", dict(status="BUILT", split="train", completed_episodes=72,
          independent_blocks=18, cortical_units=[100000], control_delivery_violations=0, refractory_violations=0,
          design_sha256=sha256_file(acquisition / "design.json"),
          episode_table_sha256=sha256_file(fields / "episodes.csv"),
          settings=dict(bin_ms=5., start_ms=0., cells_per_axis=4)))
    data = SimpleNamespace(unit_ids=np.arange(100000), unit_field_index=np.zeros(100000, dtype=np.int64),
                           field_ids=np.array(["field"]), minimum_interval_ticks=np.full(100000, 21),
                           source_identity=dict(native_library_sha256="a" * 64), dt_ms=.1, bin_ms=5.,
                           duration_ticks=60000, bin_ticks=50)
    monkeypatch.setattr(training, "NativeRenewalEpisode", lambda *a, **kw: data)
    cohort = training.TrainingCohort(acquisition, fields, training.FIXED_BUDGET)
    assert len(cohort.fit) == 56 and len(cohort.validation) == 16
    first, second = [acquisition / c["case_id"] / "report.json" for c in cases[:2]]
    first.write_bytes(second.read_bytes())
    with pytest.raises(ValueError, match="bound episode report"):
        training.TrainingCohort(acquisition, fields, training.FIXED_BUDGET)


def test_continued_budget_preserves_exact_rng_optimizer_and_full_curve(tmp_path, training_fixture):
    _, config = training_fixture
    parent = tmp_path / 'parent'
    initial = training.train(tmp_path, tmp_path, config, parent, 41)
    assert initial['completed_epochs'] == 20
    before = snapshot(parent)
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    training.train(tmp_path, tmp_path, config, whole, 41, resume_from=parent, total_epochs=24)
    training.train(tmp_path, tmp_path, config, split, 41, resume_from=parent, total_epochs=24, stop_after_epochs=2)
    assert _checkpoint(split)['next_epoch'] == 22
    training.train(tmp_path, tmp_path, config, split, 41, resume=True)
    a, b = _checkpoint(whole), _checkpoint(split)
    assert a['next_epoch'] == b['next_epoch'] == 24
    for key in ('model','optimizer','variational_rng','history','next_epoch','best','stale','best_epoch'):
        _same(a[key], b[key])
    assert _restored_loss(a) == _restored_loss(b)
    assert a['binding']['config']['maximum_epochs'] == 20 and a['effective_maximum_epochs'] == 24
    assert a['history'][:20] == _checkpoint(parent)['history']
    assert snapshot(parent) == before
    with (split / 'validation.csv').open() as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 24 * 4
    assert len({row['variational_seed'] for row in rows}) == 4


def test_legacy_parent_source_migration_is_narrow_and_ordinary_resume_stays_strict(tmp_path, training_fixture, monkeypatch):
    _, config = training_fixture
    parent = tmp_path / 'campaign' / 'renewal' / 'seed_41'
    training.train(tmp_path, tmp_path, config, parent, 41, stop_after_epochs=2)
    runtime = make_legacy_parent(parent, 'renewal', monkeypatch)
    before = snapshot(parent)
    with pytest.raises(ValueError, match='binding changed'):
        training.train(tmp_path, tmp_path, config, parent, 41, resume=True)
    assert snapshot(parent) == before
    child = tmp_path / 'child'
    report = training.train(tmp_path, tmp_path, config, child, 41, resume_from=parent, total_epochs=21, stop_after_epochs=1)
    assert report['completed_epochs'] == 3
    assert set(report['continuation']['source_migration']) == {'poisson_trajectory_training.py','renewal_trajectory_training.py'}
    assert report['continuation']['parent_checkpoint_hash_qualification'] == 'existing_parent_latest_receipt'
    assert snapshot(parent) == before
    with (runtime / 'renewal_trajectory_training.py').open('a') as stream: stream.write('\n# unregistered change\n')
    with pytest.raises(ValueError, match='pinned source revision'):
        training.train(tmp_path, tmp_path, config, tmp_path / 'rejected', 41, resume_from=parent, total_epochs=21)
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('problem', ['missing_digest','wrong_declared_digest','rng_missing','history','source','budget'])
def test_continuation_parent_receipts_rng_history_and_sources_are_enforced(problem, tmp_path, training_fixture):
    _, config = training_fixture
    parent = tmp_path / 'parent'
    training.train(tmp_path, tmp_path, config, parent, 41, stop_after_epochs=2)
    report = json.loads((parent / 'run.json').read_text())
    checkpoint = _checkpoint(parent)
    kwargs = {}
    if problem == 'missing_digest': report.pop('latest_checkpoint_sha256')
    elif problem == 'wrong_declared_digest': kwargs['parent_checkpoint_sha256'] = '0' * 64
    elif problem == 'history':
        history = json.loads((parent / 'epochs.json').read_text()); history[0]['epoch'] = 9
        (parent / 'epochs.json').write_text(json.dumps(history))
    else:
        if problem == 'rng_missing': checkpoint.pop('variational_rng')
        elif problem == 'source':
            checkpoint['binding']['observation_implementation_sha256'] = '0' * 64
            report['binding'] = checkpoint['binding']
        else: checkpoint['effective_maximum_epochs'] = 25
        torch.save(checkpoint, parent / 'latest.pt')
        report['latest_checkpoint_sha256'] = sha256_file(parent / 'latest.pt')
    (parent / 'run.json').write_text(json.dumps(report))
    before = snapshot(parent)
    with pytest.raises(ValueError):
        training.train(tmp_path, tmp_path, config, tmp_path / 'rejected', 41,
                       resume_from=parent, total_epochs=21, **kwargs)
    assert not (tmp_path / 'rejected').exists() and snapshot(parent) == before


def test_continuation_inherits_early_stop_and_selected_checkpoint(tmp_path, training_fixture, monkeypatch):
    _, config = training_fixture
    monkeypatch.setattr(training, 'evaluate_validation', lambda *a, **k: {name:1. for name in training.LOSS_TERMS})
    parent = tmp_path / 'parent'
    result = training.train(tmp_path, tmp_path, config, parent, 41)
    assert result['completed_epochs'] == 5 and result['stale_epochs'] == 4
    def no_training(*args): raise AssertionError('inherited early stop was reset')
    monkeypatch.setattr(ControlledRenewalLDS, 'sequence_loss', no_training)
    child = tmp_path / 'child'
    result = training.train(tmp_path, tmp_path, config, child, 41, resume_from=parent, total_epochs=21)
    assert result['completed_epochs'] == 5 and result['stop_reason'] == 'validation_patience'
    for key in ('model','optimizer','variational_rng','history','best','stale','best_epoch'):
        _same(_checkpoint(parent)[key], _checkpoint(child)[key])
    assert torch.load(child / 'best.pt', weights_only=True)['binding'] == _checkpoint(child)['binding']


def test_continuation_cli_forwards_registered_budget_and_parent_hash(monkeypatch, capsys):
    captured = {}
    def fake_train(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {'status':'CLI_ARGUMENTS_ONLY'}
    monkeypatch.setattr(training, 'train', fake_train)
    monkeypatch.setattr(sys, 'argv', ['trainer', '--acquisition','a','--fields','f','--config','c',
        '--output','child','--seed','43','--resume-from','parent','--total-epochs','24',
        '--parent-checkpoint-sha256','a'*64,'--stop-after-epochs','2'])
    training.main()
    assert captured['args'][-1] is False
    assert captured['kwargs'] == dict(resume_from=Path('parent'), total_epochs=24,
        parent_checkpoint_sha256='a'*64, stop_after_epochs=2)
    assert json.loads(capsys.readouterr().out)['status'] == 'CLI_ARGUMENTS_ONLY'
