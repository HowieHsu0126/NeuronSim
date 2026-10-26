"""Verify source selection and construction identity without executing rate science."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from . import v1_lgn_runtime as runtime
from .provenance import sha256_file
from .v1_lgn_source import CONSTRUCTION_FILES


def test_gray_task_preserves_source_grid_clock_and_restores_constructor():
    class Grating:
        def __init__(self, rows, cols):
            self.shape = (rows, cols)
        def create_movie(self, contrast=1.0, t_max=3.0):
            assert 0 < contrast <= 1
            return SimpleNamespace(data=np.full((4, *self.shape), contrast),
                                   t_range=np.linspace(0, t_max, 4))
    functions = SimpleNamespace(GratingMovie=Grating)
    original = runtime.create_task_movie(functions, dict(contrast=0.8, t_max=3.0))
    gray = runtime.create_task_movie(functions, dict(contrast=0.0, t_max=3.0))
    assert gray.data.shape == original.data.shape
    assert np.array_equal(gray.t_range, original.t_range)
    assert np.all(gray.data == 0) and np.all(original.data == 0.8)
    assert functions.GratingMovie is Grating
    with pytest.raises(AssertionError):
        runtime.create_task_movie(functions, dict(contrast=-0.1, t_max=3.0))
    assert functions.GratingMovie is Grating


def _binding():
    return dict(delivery_sha256='a'*64, mode='canary',
                files={name: dict(bytes=1, sha256='b'*64) for name in CONSTRUCTION_FILES})


def _case(tmp_path, monkeypatch, coverage='model_representatives'):
    config = dict(scope='development_source_runtime_qualification', unit_coverage=coverage,
        stimulus=dict(duration_s=.002, gray_screen_s=0., spatial_frequency_cycles_per_degree=.04,
                      temporal_frequency_hz=2., direction_deg=315., contrast_percent=80.))
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(config))
    cells = pd.DataFrame(dict(model_id=np.asarray([f'model_{i}' for i in range(14)])[np.arange(17400)%14],
                              x=np.arange(17400)+.125, y=np.arange(17400)+.25), index=np.arange(17400))
    models = pd.DataFrame(dict(model_id=[f'model_{i}' for i in range(14)]))
    binding = _binding()
    def calculate(nodes, kind, prefix, *parameters):
        assert kind == 'gratings'
        records = nodes.nodes()
        assert records[0][1]['position'] == (.125, .25)
        with h5py.File(prefix+'_f_tot.h5', 'w') as handle:
            handle['time'] = np.arange(3)/1000.
            handle['firing_rates_Hz'] = np.asarray([[float(source_id % 10)]*3 for source_id, _ in records])
    calculate = Mock(side_effect=calculate)
    monkeypatch.setattr(runtime, 'load_prepared_runtime', lambda _: (SimpleNamespace(calculate_firing_rate=calculate), None))
    monkeypatch.setattr(runtime, 'read_input_construction_cells', lambda _: (cells, models, binding))
    return path, config, cells, models, binding, calculate


@pytest.mark.parametrize('coverage, expected', [('model_representatives', 14), ('all_source_units', 17400)])
def test_constructed_source_coverage_and_binding_are_reported(tmp_path, monkeypatch, coverage, expected):
    path, _, _, _, binding, _ = _case(tmp_path, monkeypatch, coverage)
    report = runtime.probe_runtime(tmp_path/'runtime', None, path, tmp_path/'probe',
                                   input_construction_archive=tmp_path/'construction')
    assert report['input_construction_binding'] == binding
    assert report['source_unit_ids'] == list(range(expected))
    assert report['source_units'] == expected and report['rate_shape'] == [expected, 3]
    assert report['source_models'][0] == 'model_0'
    assert report['source_fidelity_qualified'] is False and report['production_episodes'] == 0


def test_legacy_positional_source_interface_still_works(tmp_path, monkeypatch):
    path, _, cells, models, _, _ = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, 'delivered_file', lambda archive, name: archive/name)
    legacy = Mock(return_value=(cells, models)); monkeypatch.setattr(runtime, 'read_source_cells', legacy)
    report = runtime.probe_runtime(tmp_path/'runtime', tmp_path/'legacy', path, tmp_path/'probe')
    assert legacy.call_count == 1 and report['source_units'] == 14
    assert 'input_construction_binding' not in report


@pytest.mark.parametrize('reference_kind', ['changed_bkg', 'legacy'])
def test_reference_requires_same_construction_before_rate_execution(tmp_path, monkeypatch, reference_kind):
    path, config, _, _, binding, calculate = _case(tmp_path, monkeypatch)
    reference = dict(status='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT', config=config,
                     source_unit_ids=list(range(14)))
    if reference_kind == 'changed_bkg':
        changed = deepcopy(binding); changed['files']['primary/glif/bkg_nodes.h5']['sha256'] = 'c'*64
        reference['input_construction_binding'] = changed
    archive = tmp_path/'reference'; (archive/'probe').mkdir(parents=True)
    target = archive/'probe/probe.json'; target.write_text(json.dumps(reference))
    (archive/'delivery.json').write_text(json.dumps(dict(exit_code=0, files=[dict(path='probe/probe.json',
        bytes=target.stat().st_size, sha256=sha256_file(target))])))
    with pytest.raises(ValueError, match='REFERENCE_INPUT_CONSTRUCTION_DIFFERS'):
        runtime.probe_runtime(tmp_path/'runtime', None, path, tmp_path/'probe', archive,
                              input_construction_archive=tmp_path/'construction')
    calculate.assert_not_called()


@pytest.mark.parametrize('source_option', ['--tables-archive', '--input-construction'])
def test_probe_cli_supports_each_explicit_source(monkeypatch, source_option):
    call = Mock(); monkeypatch.setattr(runtime, 'probe_runtime', call)
    runtime.main(['probe', '--runtime', 'runtime', '--config', 'config.yaml', '--output', 'out', source_option, 'source'])
    assert call.call_count == 1
    assert call.call_args.kwargs['input_construction_archive'] == (Path('source') if source_option == '--input-construction' else None)
    assert call.call_args.args[1] == (Path('source') if source_option == '--tables-archive' else None)


@pytest.mark.parametrize('options', [[], ['--tables-archive', 'old', '--input-construction', 'constructed']])
def test_probe_cli_requires_exactly_one_source(options):
    with pytest.raises(SystemExit) as error:
        runtime.main(['probe', '--runtime', 'runtime', '--config', 'config.yaml', '--output', 'out', *options])
    assert error.value.code == 2
