import copy
import unittest
import sys
import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
import yaml

from . import poisson_trajectory_training as training
from .provenance import sha256_file

from .poisson_trajectory_training import partition


class TrainingPartitionTests(unittest.TestCase):
    def test_complete_blocks_partition_and_no_calibration_or_test_intake(self):
        design = dict(acquired_split="train", cases=[dict(case_id=f"b{b}_t{t}", block=b,
                      split="train", family="trajectory") for b in range(3) for t in range(4)])
        config = dict(fit_blocks=[0, 1], validation_blocks=[2], training_seeds=[41, 42, 43])
        fit, validation = partition(design, config)
        self.assertEqual(len(fit), 8)
        self.assertEqual(len(validation), 4)
        self.assertEqual({row["block"] for row in validation}, {2})
        for change in [dict(fit_blocks=[0, 2]), dict(validation_blocks=[1, 2]), dict(validation_blocks=[])]:
            with self.assertRaises(ValueError):
                partition(design, dict(config, **change))
        wrong = copy.deepcopy(design); wrong["acquired_split"] = "test"
        with self.assertRaisesRegex(ValueError, "only the training cohort"):
            partition(wrong, config)


# Continuation tests use small explicit observation-boundary fixtures. They do
# not alter the production training configuration or represent scientific data.


class TinyPoissonEpisode:
    def __init__(self, episode, fields):
        self.unit_ids = np.array([3, 7, 12])
        self.duration_ticks, self.dt_ms = 60000, .1
        self.offset = int(episode.name.rsplit('_', 1)[-1])

    def window(self, start, stop):
        counts = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        if self.offset == 1:
            counts = counts[::-1].copy()
        return counts, np.array([[0.], [1.], [0.], [1.]], dtype=np.float32)


@pytest.fixture
def poisson_fixture(tmp_path, monkeypatch):
    acquisition, fields = tmp_path / 'acquisition', tmp_path / 'fields'
    acquisition.mkdir(); fields.mkdir()
    cases = [dict(case_id=f'case_{b}', block=b, split='train', family='trajectory') for b in range(3)]
    (acquisition / 'design.json').write_text(json.dumps(dict(acquired_split='train', cases=cases)))
    (fields / 'dataset.json').write_text(json.dumps(dict(status='BUILT', split='train', completed_episodes=3,
        design_sha256=sha256_file(acquisition / 'design.json'), settings=dict(bin_ms=5.))))
    for case in cases:
        folder = fields / case['case_id']; folder.mkdir()
        np.savez(folder / 'unit_rates.npz', unit_ids=np.array([3, 7, 12]), spike_count=np.array([4, 3, 2]))
        (folder / 'summary.json').write_text(json.dumps(dict(window_start_tick=0, window_stop_tick=60000,
            dt_ms=.1, artifacts={'unit_rates.npz': sha256_file(folder / 'unit_rates.npz')})))
    config = dict(training_seeds=[41,42,43], fit_blocks=[0,1], validation_blocks=[2], latent_size=2,
        encoder_size=3, bin_ms=5., learning_rate=.001, maximum_epochs=20, early_stopping_patience=4,
        initialization_pseudocount=.5, gradient_norm_limit=100., training_sequence='complete_six_second_episode',
        device='cpu', cpu_threads=1)
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(training, 'NativeEpisodeCounts', TinyPoissonEpisode)
    return acquisition, fields, config, path


def state(path):
    return torch.load(path / 'latest.pt', map_location='cpu', weights_only=True)


def same_state(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): same_state(a, b)
    else:
        assert left == right


def snapshot(folder):
    return {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob('*') if p.is_file()}


def make_legacy_parent(parent, model_type, monkeypatch):
    """Synthetic legacy-format/source-revision fixture; not archived execution."""
    sources = training._trainer_sources(model_type)
    runtime = parent.parent.parent / 'runtime' / 'Libs' / 'population_dynamics'
    runtime.mkdir(parents=True, exist_ok=True)
    for name in sources:
        shutil.copyfile(Path(training.__file__).with_name(name), runtime / name)
        if name in training.LEGACY_TRAINER_SOURCES:
            with (runtime / name).open('a') as stream:
                stream.write('\n# synthetic legacy source-version fixture\n')
            monkeypatch.setitem(training.LEGACY_TRAINER_SOURCES, name, sha256_file(runtime / name))
    report = json.loads((parent / 'run.json').read_text())
    for name in ('latest.pt', 'best.pt'):
        checkpoint = torch.load(parent / name, map_location='cpu', weights_only=True)
        for key in ('trainer_sources', 'effective_maximum_epochs', 'continuation'):
            checkpoint.pop(key)
        if model_type == 'renewal':
            checkpoint['binding']['trainer_implementation_sha256'] = training.LEGACY_TRAINER_SOURCES['renewal_trajectory_training.py']
            checkpoint['binding']['training_utilities_implementation_sha256'] = training.LEGACY_TRAINER_SOURCES['poisson_trajectory_training.py']
        else:
            checkpoint.pop('best_epoch', None)
        torch.save(checkpoint, parent / name)
        report['binding'] = checkpoint['binding']
    for key in ('trainer_sources', 'effective_maximum_epochs', 'continuation'):
        report.pop(key)
    if model_type == 'poisson':
        report.pop('latest_checkpoint_sha256')
        report.pop('best_epoch')
    else:
        report['latest_checkpoint_sha256'] = sha256_file(parent / 'latest.pt')
    report['best_checkpoint_sha256'] = sha256_file(parent / 'best.pt')
    (parent / 'run.json').write_text(json.dumps(report))
    return runtime


def test_poisson_split_and_continuation_are_exact_with_no_training_rng_consumption(tmp_path, poisson_fixture):
    acquisition, fields, _, config = poisson_fixture
    parent, split_base = tmp_path / 'parent', tmp_path / 'split_base'
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    training.train(acquisition, fields, config, parent, 41)
    training.train(acquisition, fields, config, split_base, 41, stop_after_epochs=1)
    training.train(acquisition, fields, config, split_base, 41, resume=True)
    for key in ('model','optimizer','history','next_epoch','best','stale','best_epoch'):
        same_state(state(parent)[key], state(split_base)[key])
    assert state(parent)['next_epoch'] == 20
    before = snapshot(parent)
    whole, divided = tmp_path / 'whole', tmp_path / 'divided'
    training.train(acquisition, fields, config, whole, 41, resume_from=parent, total_epochs=24)
    training.train(acquisition, fields, config, divided, 41, resume_from=parent, total_epochs=24, stop_after_epochs=2)
    assert state(divided)['next_epoch'] == 22
    training.train(acquisition, fields, config, divided, 41, resume=True)
    left, right = state(whole), state(divided)
    assert left['next_epoch'] == right['next_epoch'] == 24
    for key in ('model','optimizer','history','next_epoch','best','stale','best_epoch'):
        same_state(left[key], right[key])
    assert left['binding']['config']['maximum_epochs'] == 20
    assert left['effective_maximum_epochs'] == 24
    assert left['history'][:20] == state(parent)['history']
    assert snapshot(parent) == before
    assert random.getstate() == python_rng
    same_state(list(np.random.get_state()[2:]), list(numpy_rng[2:]))
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    with (divided / 'optimization.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    for epoch in range(24):
        expected = ['case_0','case_1']; random.Random(41 * 100000 + epoch).shuffle(expected)
        assert [r['case_id'] for r in rows if int(r['epoch']) == epoch] == expected


def test_poisson_zero_work_resume_retains_completed_epoch_metadata(tmp_path, poisson_fixture):
    acquisition, fields, _, config = poisson_fixture
    output = tmp_path / 'run'
    training.train(acquisition, fields, config, output, 41)
    before = state(output)
    report = training.train(acquisition, fields, config, output, 41, resume=True)
    assert report['completed_epochs'] == before['next_epoch'] == 20
    assert report['best_epoch'] == before['best_epoch']
    assert report['stale_epochs'] == before['stale']
    assert json.loads((output / 'epochs.json').read_text()) == before['history']
    same_state(before, state(output))


def test_poisson_legacy_parent_requires_prospective_checksum_and_keeps_parent_immutable(tmp_path, poisson_fixture, monkeypatch):
    acquisition, fields, _, config = poisson_fixture
    parent = tmp_path / 'campaign' / 'poisson' / 'seed_41'
    training.train(acquisition, fields, config, parent, 41, stop_after_epochs=2)
    runtime = make_legacy_parent(parent, 'poisson', monkeypatch)
    before = snapshot(parent)
    with pytest.raises(ValueError, match='prospectively registered'):
        training.train(acquisition, fields, config, tmp_path / 'missing', 41, resume_from=parent, total_epochs=21)
    assert not (tmp_path / 'missing').exists()
    report = training.train(acquisition, fields, config, tmp_path / 'child', 41, resume_from=parent,
        total_epochs=21, parent_checkpoint_sha256=sha256_file(parent / 'latest.pt'), stop_after_epochs=1)
    assert report['completed_epochs'] == 3
    assert report['continuation']['parent_checkpoint_hash_qualification'].startswith('prospectively_bound_parent_snapshot')
    assert set(report['continuation']['source_migration']) == {'poisson_trajectory_training.py'}
    assert snapshot(parent) == before
    with (runtime / 'native_count_windows.py').open('a') as stream: stream.write('\n# changed numerical source\n')
    with pytest.raises(ValueError, match='pinned source revision'):
        training.train(acquisition, fields, config, tmp_path / 'bad_source', 41, resume_from=parent,
            total_epochs=21, parent_checkpoint_sha256=sha256_file(parent / 'latest.pt'))
    assert not (tmp_path / 'bad_source').exists()


@pytest.mark.parametrize('problem', ['latest_digest','best_digest','history','report_seed','config','numerical_source','unregistered_trainer'])
def test_poisson_parent_tampering_or_mismatch_rejected_before_child_write(problem, tmp_path, poisson_fixture):
    acquisition, fields, values, config = poisson_fixture
    parent = tmp_path / 'parent'
    training.train(acquisition, fields, config, parent, 41, stop_after_epochs=2)
    if problem in ('latest_digest','best_digest'):
        with (parent / ('latest.pt' if problem == 'latest_digest' else 'best.pt')).open('ab') as stream: stream.write(b'changed')
    elif problem == 'history':
        history = json.loads((parent / 'epochs.json').read_text()); history[0]['validation']['loss'] += 1
        (parent / 'epochs.json').write_text(json.dumps(history))
    elif problem == 'config':
        values['learning_rate'] *= 2; config.write_text(yaml.safe_dump(values))
    else:
        report = json.loads((parent / 'run.json').read_text())
        checkpoint = state(parent)
        if problem == 'report_seed': report['training_seed'] = 42
        elif problem == 'numerical_source':
            checkpoint['binding']['model_implementation_sha256'] = '0' * 64
            report['binding'] = checkpoint['binding']
        else:
            checkpoint['trainer_sources']['poisson_trajectory_training.py'] = '0' * 64
            report['trainer_sources'] = checkpoint['trainer_sources']
        torch.save(checkpoint, parent / 'latest.pt')
        report['latest_checkpoint_sha256'] = sha256_file(parent / 'latest.pt')
        (parent / 'run.json').write_text(json.dumps(report))
    before = snapshot(parent)
    with pytest.raises(ValueError):
        training.train(acquisition, fields, config, tmp_path / 'rejected', 41, resume_from=parent, total_epochs=21)
    assert not (tmp_path / 'rejected').exists() and snapshot(parent) == before


def test_poisson_ordinary_resume_budget_is_strict_and_continuation_requires_new_path(tmp_path, poisson_fixture):
    acquisition, fields, values, config = poisson_fixture
    parent = tmp_path / 'parent'
    training.train(acquisition, fields, config, parent, 41, stop_after_epochs=2)
    before = snapshot(parent)
    for total in (2,20):
        with pytest.raises(ValueError, match='registered ceilings'):
            training.train(acquisition, fields, config, tmp_path / f'bad_{total}', 41, resume_from=parent, total_epochs=total)
    with pytest.raises(ValueError, match='must not modify'):
        training.train(acquisition, fields, config, parent / 'nested', 41, resume_from=parent, total_epochs=21)
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(FileExistsError):
        training.train(acquisition, fields, config, existing, 41, resume_from=parent, total_epochs=21)
    with pytest.raises(ValueError, match='mutually exclusive'):
        training.train(acquisition, fields, config, tmp_path / 'both', 41, resume=True, resume_from=parent, total_epochs=21)
    values['maximum_epochs'] = 21; config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='binding changed'):
        training.train(acquisition, fields, config, parent, 41, resume=True)
    assert snapshot(parent) == before


def test_poisson_continuation_inherits_early_stop_without_resetting_best(tmp_path, poisson_fixture, monkeypatch):
    acquisition, fields, _, config = poisson_fixture
    original = training.ControlledPoissonLDS.sequence_loss
    def plateau(self, *args):
        terms = original(self, *args)
        if not torch.is_grad_enabled(): terms['loss'] = terms['loss'].new_tensor(1.)
        return terms
    monkeypatch.setattr(training.ControlledPoissonLDS, 'sequence_loss', plateau)
    parent = tmp_path / 'parent'
    result = training.train(acquisition, fields, config, parent, 41)
    assert result['completed_epochs'] == 5 and result['stale_epochs'] == 4
    def no_update(*args): raise AssertionError('inherited early stop was reset')
    monkeypatch.setattr(training.ControlledPoissonLDS, 'sequence_loss', no_update)
    child = tmp_path / 'child'
    result = training.train(acquisition, fields, config, child, 41, resume_from=parent, total_epochs=21)
    assert result['completed_epochs'] == 5 and result['stop_reason'] == 'validation_patience'
    for key in ('model','optimizer','history','best','stale','best_epoch'):
        same_state(state(parent)[key], state(child)[key])
    assert torch.load(child / 'best.pt', weights_only=True)['binding'] == state(child)['binding']


def test_continuation_cli_forwards_registered_budget_and_parent_hash(monkeypatch, capsys):
    captured = {}
    def fake_train(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {'status':'CLI_ARGUMENTS_ONLY'}
    monkeypatch.setattr(training, 'train', fake_train)
    monkeypatch.setattr(sys, 'argv', ['trainer', '--acquisition','a','--fields','f','--config','c',
        '--output','child','--seed','41','--resume-from','parent','--total-epochs','24',
        '--parent-checkpoint-sha256','a'*64,'--stop-after-epochs','2'])
    training.main()
    assert captured['args'][-1] is False
    assert captured['kwargs'] == dict(resume_from=Path('parent'), total_epochs=24,
        parent_checkpoint_sha256='a'*64, stop_after_epochs=2)
    assert json.loads(capsys.readouterr().out)['status'] == 'CLI_ARGUMENTS_ONLY'


def test_ordinary_resume_cannot_change_registered_child_budget(tmp_path, poisson_fixture):
    acquisition, fields, _, config = poisson_fixture
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    training.train(acquisition, fields, config, parent, 41, stop_after_epochs=2)
    training.train(acquisition, fields, config, child, 41, resume_from=parent, total_epochs=21, stop_after_epochs=1)
    checkpoint = state(child)
    checkpoint['effective_maximum_epochs'] = 22
    checkpoint['continuation']['total_epochs'] = 22
    torch.save(checkpoint, child / 'latest.pt')
    before = snapshot(child)
    with pytest.raises(ValueError, match='registered budget changed'):
        training.train(acquisition, fields, config, child, 41, resume=True)
    assert snapshot(child) == before


if __name__ == '__main__':
    unittest.main()
