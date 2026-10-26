"""Matched deterministic-mean estimands, causal prefixes and replay rejection."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from . import braid_trajectory_evaluation as evaluation


def test_mean_scores_use_identical_full_unit_and_field_denominators():
    means = np.array([[1., 2., 4.], [3., 1., 0.]])
    truth = np.array([[0, 2, 3], [1, 2, 0]])
    index = np.array([0, 1, 0]); units = np.array([2, 1]); sd = np.array([2., 4.])
    metrics, fields, actual = evaluation.mean_scores(means, truth, index, units, sd, .5)
    np.testing.assert_array_equal(fields, [[5., 2.], [3., 1.]])
    np.testing.assert_array_equal(actual, [[3, 2], [1, 2]])
    assert metrics['unit_bin_mean_mse'] == pytest.approx(7 / 6)
    assert metrics['fitting_standardized_field_rate_mse'] == pytest.approx(.5625)
    with pytest.raises(ValueError): evaluation.mean_scores(means, truth, index, np.array([1, 1]), sd, .5)
    with pytest.raises(ValueError): evaluation.mean_scores(means, truth + .1, index, units, sd, .5)


class TinyModel:
    unit_ids = (1, 2)
    bin_seconds = .005
    def __init__(self): self.prefix_lengths = []
    def infer_prefix(self, history, metadata):
        self.prefix_lengths.append(history['counts'].shape[1])
        return SimpleNamespace(boundary=tf.constant(history['counts'][:, -1], dtype=tf.float32))
    def forecast(self, state, controls):
        return tf.constant(np.broadcast_to(state.boundary.numpy()[:, None], (1, len(controls[0]), 2)).copy())


class TinyCohort:
    bin_seconds = .005
    unit_ids = np.array([1, 2]); unit_field_index = np.array([0, 1]); field_ids = np.array(['a', 'b'])
    binding = {'fixture': 'separate observed prefix and complete future'}
    validation = [dict(case_id=f'case_{i}', block=14 + i // 4) for i in range(16)]
    def window(self, case):
        counts = np.zeros((1200, 2), dtype=np.float32)
        counts[:, 0] = np.arange(1200) % 3
        return counts, np.zeros((1200, 1), dtype=np.float32)


def test_forecast_uses_all_observed_bins_for_each_matched_origin():
    model = TinyModel(); cohort = TinyCohort()
    for origin, bins in [(1000., 200), (3000., 600), (5000., 1000)]:
        rows, _, means, truth, costs = evaluation._forecast(model, cohort, cohort.validation[0], origin, np.ones(2), np.ones(2))
        assert model.prefix_lengths[-1] == bins
        assert means.shape == truth.shape == (100, 2)
        assert len(rows) == 3 and costs['generated_unit_mean_values'] == 200
        assert set(row['horizon_ms'] for row in rows) == {20., 80., 500.}


def test_complete_evaluation_replay_and_changed_forecast_rejection(tmp_path, monkeypatch):
    model, cohort = TinyModel(), TinyCohort()
    model.native = SimpleNamespace(model=SimpleNamespace(count_params=lambda: 10))
    state = dict(elapsed_seconds=1.)
    monkeypatch.setattr(evaluation, '_load', lambda *a: (model, cohort, state, np.ones(2), np.ones(2), cohort.field_ids))
    training = tmp_path / 'training'; training.mkdir(); (training / 'committed.json').write_text('{}')
    config = tmp_path / 'config.yaml'; config.write_text('fixture: true')
    scale = tmp_path / 'scale.npz'; np.savez(scale, fixture=np.ones(2))
    output = tmp_path / 'evaluation'
    report = evaluation.evaluate(tmp_path, tmp_path, training, config, scale, output, 41)
    assert report['score_rows'] == 144 and report['origins'] == 48
    replay = evaluation.evaluate(tmp_path, tmp_path, training, config, scale, output, 41, replay=True)
    assert replay['status'] == 'PASS' and replay['verified_score_rows'] == 144
    forecast = output / 'case_0_origin_1000.npz'
    forecast.write_bytes(forecast.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='saved forecast changed'):
        evaluation.evaluate(tmp_path, tmp_path, training, config, scale, output, 41, replay=True)
