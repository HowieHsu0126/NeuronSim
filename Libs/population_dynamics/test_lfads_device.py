"""Explicit execution placement and portable LFADS checkpoint metadata.

CUDA success is validated on a separately allocated GPU. The metadata fixture
below tests load policy only; it is not evidence of GPU execution or equivalence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import yaml

from . import lfads
from .lfads import LFADSConditionalSimulator, LFADSRandomStreams


CONFIG = Path(__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml'


def spec():
    return {**yaml.safe_load(CONFIG.read_text())['lfads_baseline_check']['model'],
            'keep_prob': .65, 'controller_input_lag': 2,
            'implicit_identity_readin': True, 'gradient_aggregation': 'tree'}


def create(**kwargs):
    return LFADSConditionalSimulator([12, 4, 9], 1, 3, 3, .005, spec(),
                                     initialization_seed=41, **kwargs)


def arrays():
    counts = (np.arange(2 * 6 * 3).reshape(2, 6, 3) % 4).astype(np.int64)
    stimulus = np.linspace(0, .8, 12, dtype=np.float32).reshape(2, 6, 1)
    return counts, stimulus


def streams():
    return LFADSRandomStreams.from_seeds(51, 52, 53)


def assert_values_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def assert_cpu_placement(model):
    assert model.device == 'cpu'
    assert model.execution_device['requested'] == 'cpu'
    assert model.execution_device['device_type'] == 'CPU'
    assert model.execution_device['tensorflow_device'].endswith('/device:CPU:0')
    assert model.execution_device['tensorflow_version'] == tf.__version__
    for view in (model._full, model._prefix):
        assert view.execution_device == model.execution_device
        config = view.session._config
        assert config.device_count['GPU'] == 0
        assert not config.allow_soft_placement
        assert config.intra_op_parallelism_threads == config.inter_op_parallelism_threads == 2
        assert all(entry.device_type != 'GPU' for entry in view.session.list_devices())
        assert all(variable.device.lower().endswith('/device:cpu:0')
                   for variable in view.variables.values())


def test_default_and_explicit_cpu_match_and_preserve_global_state(tmp_path):
    eager_before = tf.executing_eagerly()
    global_graph = tf.get_default_graph()
    global_seed = global_graph.seed
    visible_before = os.environ.get('CUDA_VISIBLE_DEVICES')
    counts, stimulus = arrays()
    with create() as default, create(device='cpu') as explicit:
        assert_cpu_placement(default)
        assert_cpu_placement(explicit)
        assert default._full.randomness.sites == explicit._full.randomness.sites
        assert_values_equal(default._full.values(), explicit._full.values())
        assert default.fit_step(counts, stimulus) == explicit.fit_step(counts, stimulus)
        assert_values_equal(default._full.values(), explicit._full.values())
        history = {'counts': counts[:, :3], 'stimulus': stimulus[:, :3]}
        state = default.infer_state(history)
        other_state = explicit.infer_state(history)
        weights = np.ones((2, 3, 3), dtype=np.float32)
        before = set(default._full.graph.get_operations())
        first = default.expected_count_gradient(state, stimulus[:, 3:], weights, streams(), 2)
        second = explicit.expected_count_gradient(other_state, stimulus[:, 3:], weights, streams(), 2)
        np.testing.assert_array_equal(first, second)
        added = set(default._full.graph.get_operations()) - before
        assert added and all(op.device.lower().endswith('/device:cpu:0') for op in added)
        checkpoint = tmp_path / 'checkpoint'
        default.save(checkpoint)
        record = json.loads((checkpoint / 'model.json').read_text())
        assert record['execution_device'] == default.execution_device
        with LFADSConditionalSimulator.load(checkpoint, device='cpu') as restored:
            assert_cpu_placement(restored)
            assert restored.checkpoint_execution_device == record['execution_device']
            assert not restored.loaded_for_inference
            assert restored.training_updates == default.training_updates == 1
            assert_values_equal(default._full.values(), restored._full.values())
            for _ in range(2):
                assert default.fit_step(counts, stimulus) == restored.fit_step(counts, stimulus)
                assert_values_equal(default._full.values(), restored._full.values())
    assert tf.executing_eagerly() == eager_before
    assert tf.get_default_graph() is global_graph
    assert global_graph.seed == global_seed
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == visible_before


@pytest.fixture(scope='module')
def saved_checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp('cpu_checkpoint') / 'model'
    with create() as model:
        model.fit_step(*arrays())
        model.save(folder)
    return folder


@pytest.mark.parametrize('device', [None, True, 0, 'gpu', 'cuda:0', 'CPU', {}, []])
def test_invalid_device_rejected_before_graph_creation(device, saved_checkpoint, monkeypatch):
    def graph_must_not_start(*args, **kwargs):
        raise AssertionError('invalid device started graph construction')
    monkeypatch.setattr(lfads, '_GraphView', graph_must_not_start)
    with pytest.raises(ValueError, match='LFADS_DEVICE_MUST_BE_CPU_OR_CUDA'):
        create(device=device)
    with pytest.raises(ValueError, match='LFADS_DEVICE_MUST_BE_CPU_OR_CUDA'):
        LFADSConditionalSimulator.load(saved_checkpoint, device=device)


@pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES') != '',
                    reason='requires the explicit CPU-only helper environment')
def test_requested_cuda_with_no_visible_gpu_fails_without_cpu_fallback(saved_checkpoint, monkeypatch):
    captured = []
    actual_session = tf.Session
    def observed_session(*args, **kwargs):
        captured.append(kwargs['config'])
        return actual_session(*args, **kwargs)
    def source_must_not_start(*args, **kwargs):
        raise AssertionError('unavailable CUDA reached source model construction')
    monkeypatch.setattr(lfads.tf, 'Session', observed_session)
    monkeypatch.setattr(lfads, 'SourceLFADS', source_must_not_start)
    for construct in (lambda: create(device='cuda'),
                      lambda: LFADSConditionalSimulator.load(saved_checkpoint, device='cuda')):
        with pytest.raises(RuntimeError, match='LFADS_CUDA_REQUESTED_BUT_NO_VISIBLE_GPU'):
            construct()
    assert len(captured) == 2
    for config in captured:
        assert config.device_count['GPU'] == 1
        assert config.gpu_options.allow_growth
        assert not config.allow_soft_placement
        assert config.intra_op_parallelism_threads == config.inter_op_parallelism_threads == 2


@pytest.mark.parametrize('variant', ['legacy_v1', 'legacy_v2', 'cuda_metadata_only'])
def test_portable_load_uses_requested_device_not_checkpoint_metadata(variant, saved_checkpoint, tmp_path):
    checkpoint = tmp_path / 'fixture'
    shutil.copytree(saved_checkpoint, checkpoint)
    record = json.loads((checkpoint / 'model.json').read_text())
    if variant == 'cuda_metadata_only':
        # Deliberately synthetic provenance fixture: no GPU result is claimed.
        record['execution_device'] = {'requested': 'cuda', 'device_type': 'GPU',
            'tensorflow_device': '/job:localhost/replica:0/task:0/device:GPU:0',
            'tensorflow_version': tf.__version__}
    else:
        record.pop('execution_device')
        if variant == 'legacy_v1':
            record['schema'] = 'neural_population_lfads_conditional_v1'
            record['training_rng_resume_supported'] = False
            for field in ('training_rng_schema', 'training_updates', 'training_random_sites',
                          'training_history_bins', 'training_horizon_bins'):
                record.pop(field)
    (checkpoint / 'model.json').write_text(json.dumps(record))
    with LFADSConditionalSimulator.load(checkpoint) as model:
        assert_cpu_placement(model)
        assert model.checkpoint_execution_device == record.get('execution_device')
        assert model.loaded_for_inference == (variant == 'legacy_v1')
        assert model.training_history_bins == model.training_horizon_bins == 3
        with np.load(checkpoint / 'weights.npz', allow_pickle=False) as source:
            assert_values_equal(dict(source), model._full.values())
        resaved = tmp_path / 'resaved'
        model.save(resaved)
        assert json.loads((resaved / 'model.json').read_text())['execution_device'] == model.execution_device


def test_explicit_device_preserves_changed_length_inference_boundary(saved_checkpoint, tmp_path):
    with LFADSConditionalSimulator.load(saved_checkpoint, device='cpu', history_bins=4, horizon_bins=2) as model:
        assert_cpu_placement(model)
        assert model.loaded_for_inference
        assert model.training_history_bins == model.training_horizon_bins == 3
        assert (model.history_bins, model.horizon_bins) == (4, 2)
        with pytest.raises(ValueError, match='CHECKPOINT_IS_INFERENCE_ONLY'):
            model.fit_step(*arrays())
        checkpoint = tmp_path / 'changed_lengths'
        model.save(checkpoint)
        record = json.loads((checkpoint / 'model.json').read_text())
        assert not record['training_rng_resume_supported']
        assert record['execution_device']['requested'] == 'cpu'


@pytest.mark.skipif(os.environ.get('NEURONSIM_LFADS_CUDA_TEST') != '1',
                    reason='requires an explicitly allocated CUDA validation run')
@pytest.mark.parametrize('grappler_policy', ['default', 'off'])
def test_cuda_update_and_portable_checkpoint(tmp_path, grappler_policy):
    # Freeze cross-backend tolerances. The subprocess must select full FP32;
    # do not mutate TensorFlow global precision policy from the adapter.
    assert os.environ.get('NVIDIA_TF32_OVERRIDE') == '0'
    counts, stimulus = arrays()
    checkpoint = tmp_path / 'cpu_checkpoint'
    history = dict(counts=counts[:, :3], stimulus=stimulus[:, :3])
    weights = np.ones((2, 3, 3), dtype=np.float32)
    with LFADSConditionalSimulator([12, 4, 9], 1, 3, 3, .005,
            {**spec(), 'grappler_optimization': grappler_policy},
            initialization_seed=41, device='cpu') as cpu:
        cpu.fit_step(counts, stimulus)
        cpu.save(checkpoint)
        before = cpu._full.values()
        cpu_terms = cpu.fit_step(counts, stimulus)
        cpu_values = cpu._full.values()
        cpu_gradient = cpu.expected_count_gradient(cpu.infer_state(history),
                            stimulus[:, 3:], weights, streams(), 2)
    with LFADSConditionalSimulator.load(checkpoint, device='cuda') as gpu:
        assert gpu.execution_device['device_type'] == 'GPU'
        assert gpu.checkpoint_execution_device['device_type'] == 'CPU'
        assert_values_equal(before, gpu._full.values())
        gpu_terms = gpu.fit_step(counts, stimulus)
        for name, value in cpu_terms.items():
            np.testing.assert_allclose(gpu_terms[name], value, rtol=2e-5, atol=2e-6, err_msg=name)
        for name, value in gpu._full.values().items():
            np.testing.assert_allclose(value, cpu_values[name], rtol=2e-5, atol=2e-6, err_msg=name)
        state = gpu.infer_state(history)
        gradient = gpu.expected_count_gradient(state, stimulus[:, 3:], weights, streams(), 2)
        np.testing.assert_allclose(gradient, cpu_gradient, rtol=2e-5, atol=2e-6)
        sample = gpu.simulate(state, stimulus[:, 3:], streams(), 3)
        saved = tmp_path / 'cuda_checkpoint'
        gpu.save(saved)
        values = gpu._full.values()
    with LFADSConditionalSimulator.load(saved, device='cuda') as restored:
        repeated = restored.simulate(restored.infer_state(history), stimulus[:, 3:], streams(), 3)
        for name in ('counts', 'conditional_expected_counts', 'initial_latent_states',
                     'historical_controller_inputs', 'future_controller_inputs',
                     'boundary_generator_states', 'future_generator_states'):
            np.testing.assert_array_equal(getattr(sample, name), getattr(repeated, name))
    with LFADSConditionalSimulator.load(saved, device='cpu') as restored:
        assert_values_equal(values, restored._full.values())
        assert restored.training_updates == 2



def test_grappler_option_preserves_updates_and_resume(tmp_path):
    counts, stimulus = arrays()
    off_spec = {**spec(), 'grappler_optimization': 'off'}
    with create() as original, LFADSConditionalSimulator([12, 4, 9], 1, 3, 3,
                .005, off_spec, initialization_seed=41) as unpruned:
        assert unpruned._full.session._config.graph_options.rewrite_options.disable_meta_optimizer
        assert_values_equal(original._full.values(), unpruned._full.values())
        for _ in range(3):
            first, second = original.fit_step(counts, stimulus), unpruned.fit_step(counts, stimulus)
            for key in first:
                np.testing.assert_allclose(first[key], second[key], rtol=2e-5, atol=2e-6, err_msg=key)
            for key, value in original._full.values().items():
                np.testing.assert_allclose(value, unpruned._full.values()[key], rtol=2e-5, atol=2e-6, err_msg=key)
        history = dict(counts=counts[:, :3], stimulus=stimulus[:, :3])
        before = original.simulate(original.infer_state(history), stimulus[:, 3:], streams(), 3)
        after = unpruned.simulate(unpruned.infer_state(history), stimulus[:, 3:], streams(), 3)
        np.testing.assert_allclose(before.conditional_expected_counts, after.conditional_expected_counts,
                                   rtol=2e-5, atol=2e-6)
        checkpoint = tmp_path / 'checkpoint'
        unpruned.save(checkpoint)
        with LFADSConditionalSimulator.load(checkpoint) as restored:
            assert restored.execution_device['grappler_optimization'] == 'off'
            assert unpruned.fit_step(counts, stimulus) == restored.fit_step(counts, stimulus)
            assert_values_equal(unpruned._full.values(), restored._full.values())


@pytest.mark.parametrize('policy', [None, True, 0, 'OFF', 'disabled'])
def test_invalid_grappler_optimization_rejected(policy):
    with pytest.raises(ValueError, match='LFADS_GRAPPLER_OPTIMIZATION_MUST_BE_DEFAULT_OR_OFF'):
        LFADSConditionalSimulator([12, 4, 9], 1, 3, 3, .005,
            {**spec(), 'grappler_optimization': policy}, initialization_seed=41)
