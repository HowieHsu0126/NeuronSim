"""Verified input relocation preserves the original path and rejects corruption."""
import importlib.util
from pathlib import Path

import pytest


def relocation():
    path = Path(__file__).parents[1] / 'scripts/relocate_latent_spike_diffusion_inputs.py'
    spec = importlib.util.spec_from_file_location('ldns_relocation', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_inputs(tmp_path):
    runner = relocation()
    cell = tmp_path / 'cell'
    source = cell / 'inputs/train'
    source.mkdir(parents=True)
    (cell / 'failure.json').write_text('original failure')
    payload = source / 'spikes.bin'
    payload.write_bytes(b'complete original spikes')
    rows = [dict(path='train/spikes.bin', bytes=payload.stat().st_size, sha256=runner.digest(payload))]
    return runner, cell, rows


def test_relocation_preserves_bytes_path_receipts_and_known_retry(tmp_path):
    runner, cell, rows = fixture_inputs(tmp_path)
    destination = tmp_path / 'data_inputs'
    result = runner.relocate(cell, destination, rows)
    assert result['status'] == 'RELOCATED_VERIFIED_INPUT_COPY'
    assert (cell / 'inputs').is_symlink() and (cell / 'inputs').resolve() == destination
    assert runner.digest(cell / 'inputs/train/spikes.bin') == rows[0]['sha256']
    assert (cell / 'failure.json').read_text() == 'original failure'
    assert not (cell / 'inputs.relocation_backup').exists()
    assert runner.relocate(cell, destination, rows)['status'] == 'ALREADY_RELOCATED_VERIFIED'


@pytest.mark.parametrize('mutation', ['changed', 'extra', 'unsafe', 'duplicate', 'link'])
def test_invalid_manifest_or_input_rejects_before_copy(tmp_path, mutation):
    runner, cell, rows = fixture_inputs(tmp_path)
    if mutation == 'changed':
        (cell / 'inputs/train/spikes.bin').write_bytes(b'changed')
    elif mutation == 'extra':
        (cell / 'inputs/unlisted').write_text('retain this')
    elif mutation == 'unsafe':
        rows[0]['path'] = '../failure.json'
    elif mutation == 'duplicate':
        rows.append(dict(rows[0]))
    else:
        (cell / 'inputs/link').symlink_to(cell / 'failure.json')
    destination = tmp_path / 'data_inputs'
    with pytest.raises(ValueError):
        runner.relocate(cell, destination, rows)
    assert not destination.exists() and not (cell / 'inputs').is_symlink()
    assert (cell / 'failure.json').read_text() == 'original failure'


def test_corrupted_copy_keeps_original_input_directory(tmp_path, monkeypatch):
    runner, cell, rows = fixture_inputs(tmp_path)
    def corrupt(source, target):
        target.write_bytes(b'corrupted copy')
    monkeypatch.setattr(runner.shutil, 'copy2', corrupt)
    with pytest.raises(ValueError, match='input file changed'):
        runner.relocate(cell, tmp_path / 'data_inputs', rows)
    assert not (cell / 'inputs').is_symlink()
    assert runner.digest(cell / 'inputs/train/spikes.bin') == rows[0]['sha256']


def test_existing_destination_requires_outcome_reconciliation(tmp_path):
    runner, cell, rows = fixture_inputs(tmp_path)
    destination = tmp_path / 'data_inputs'
    destination.mkdir()
    with pytest.raises(FileExistsError, match='resolve the previous outcome'):
        runner.relocate(cell, destination, rows)
    assert not (cell / 'inputs').is_symlink()


def test_path_switch_failure_restores_original_inputs(tmp_path, monkeypatch):
    runner, cell, rows = fixture_inputs(tmp_path)
    original = Path.rename
    def failed_link_switch(path, target):
        if path.name == 'inputs.relocation_link':
            raise OSError('fixture switch failed')
        return original(path, target)
    monkeypatch.setattr(Path, 'rename', failed_link_switch)
    with pytest.raises(OSError, match='switch failed'):
        runner.relocate(cell, tmp_path / 'data_inputs', rows)
    assert not (cell / 'inputs').is_symlink()
    assert runner.digest(cell / 'inputs/train/spikes.bin') == rows[0]['sha256']
    assert not (cell / 'inputs.relocation_backup').exists()


def test_insufficient_destination_storage_rejects_before_copy(tmp_path, monkeypatch):
    from collections import namedtuple
    runner, cell, rows = fixture_inputs(tmp_path)
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda path: usage(100, 99, 1))
    destination = tmp_path / 'data_inputs'
    with pytest.raises(RuntimeError, match='copy space'):
        runner.relocate(cell, destination, rows)
    assert not destination.exists() and not (cell / 'inputs').is_symlink()


def campaign_fixture(tmp_path, monkeypatch):
    """Use the frozen manifest reader and four complete, small input trees."""
    from collections import namedtuple
    import json
    import os
    import shutil
    from types import SimpleNamespace

    runner = relocation()
    repo = tmp_path / 'repo'
    preparation = tmp_path / 'preparation'
    preparation.mkdir()
    (preparation / 'preparation.json').write_text('original preparation\n')
    original_preparation = (Path(__file__).parents[2]
        / 'Output/neural_population_dynamics/baselines/latent_spike_diffusion/preparation_20261012')
    stager = runner.module(original_preparation / 'Libs/scripts/run_population_trajectory_campaign.py',
                           'ldns_relocation_fixture_stager')
    monkeypatch.setattr(runner, 'module', lambda path, name:
        SimpleNamespace(snapshot=lambda: dict(d_state_count=0))
        if path.name == 'resource_probe.py' else stager)
    cohort = repo / stager.COHORT
    cohort.mkdir(parents=True)
    rows = [dict(path=f'train/input_{index:03d}.bin', destination_bytes=len(b'native input'),
                 sha256=runner.hashlib.sha256(b'native input').hexdigest())
            for index in range(436)]
    receipt = dict(files=rows)
    receipt_path = cohort / 'delivery.json'
    receipt_path.write_text(json.dumps(receipt))
    receipt_sha = runner.digest(receipt_path)
    failed_prefix = tmp_path / 'neuronsim-ldns-611529'
    cells = [Path(str(failed_prefix) + '-' + str(seed)) for seed in (41, 42, 43)]
    cuda_cell = tmp_path / 'neuronsim-ldns-cuda-611535'
    cells.append(cuda_cell)
    gate = repo / 'Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012'
    gate.mkdir(parents=True)
    checkpoint = b'preserved CUDA checkpoint'
    (gate / 'delivery.json').write_text(json.dumps(dict(status='PASS', files=[dict(
        path='checkpoint.pt', bytes=len(checkpoint), sha256=runner.hashlib.sha256(checkpoint).hexdigest())])))
    preserved = {}
    for index, cell in enumerate(cells):
        (cell / 'inputs/train').mkdir(parents=True)
        for row in rows:
            (cell / 'inputs' / row['path']).write_bytes(b'native input')
        (cell / 'prerequisites').mkdir()
        shutil.copy2(receipt_path, cell / 'prerequisites/cohort_delivery.json')
        (cell / 'inputs.json').write_text(json.dumps(dict(delivery_sha256=receipt_sha, files=rows)))
        if index < 3:
            (cell / 'phases/train').mkdir(parents=True)
            (cell / 'phases/train/preflight.json').write_text(json.dumps(dict(
                status='BLOCKED', reasons=['ROOT_STORAGE_CRITICAL'])))
            shutil.copy2(preparation / 'preparation.json', cell / 'preparation.json')
            (cell / 'failure.json').write_text('original storage failure')
        else:
            (cell / 'results').mkdir()
            (cell / 'results/checkpoint.pt').write_bytes(checkpoint)
        preserved.update({path: path.read_bytes() for path in cell.rglob('*')
                          if path.is_file() and cell / 'inputs' not in path.parents})

    # Model the GPU03 mount/device boundary without touching real /data.
    data = tmp_path / 'data'
    data.mkdir()
    root = tmp_path / 'root'
    root.mkdir()
    real_stat = Path.stat
    def fixture_stat(path, *args, **kwargs):
        value = real_stat(path, *args, **kwargs)
        if path == root:
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path, 'stat', fixture_stat)
    monkeypatch.setattr(runner, 'Path', lambda value:
        data if str(value) == '/data' else root if str(value) == '/' else Path(value))
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda path: usage(100 * 1024**3, 0, 40 * 1024**3))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs:
        'procs memory swap io system cpu\nsi so wa\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n')
    for name, value in dict(SLURMD_NODENAME='gpu03', SLURM_JOB_PARTITION='gpu1',
                            SLURM_JOB_GPUS='0', SLURM_JOB_ID='fixture').items():
        monkeypatch.setenv(name, value)
    return runner, repo, preparation, failed_prefix, cuda_cell, data / 'retained', tmp_path / 'output', cells, preserved


def test_full_campaign_authenticates_all_four_cells_and_preserves_results(tmp_path, monkeypatch):
    runner, repo, preparation, prefix, cuda, destination, output, cells, preserved = campaign_fixture(tmp_path, monkeypatch)
    result = runner.run(repo, preparation, prefix, cuda, destination, output)
    assert result['status'] == 'PASS_AUTHENTICATED_INPUT_RELOCATION'
    assert len(result['cells']) == 4 and all(row['files'] == 436 for row in result['cells'])
    assert result['training_started'] is False and result['training_resource_admitted'] is False
    assert result['scientific_protocol_changed'] is False
    for cell in cells:
        assert (cell / 'inputs').is_symlink()
        assert (cell / 'inputs').resolve() == destination / cell.name
        assert (cell / 'inputs/train/input_435.bin').read_bytes() == b'native input'
    assert all(path.read_bytes() == value for path, value in preserved.items())


def test_full_campaign_rejects_last_cell_corruption_before_any_relocation(tmp_path, monkeypatch):
    import json
    runner, repo, preparation, prefix, cuda, destination, output, cells, preserved = campaign_fixture(tmp_path, monkeypatch)
    (cuda / 'inputs/train/input_435.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='input file changed'):
        runner.run(repo, preparation, prefix, cuda, destination, output)
    assert not destination.exists()
    assert all(not (cell / 'inputs').is_symlink() for cell in cells)
    assert all(path.read_bytes() == value for path, value in preserved.items())
    result = json.loads((output / 'relocation.json').read_text())
    assert result['status'] == 'INCOMPLETE_LDNS_INPUT_RELOCATION' and result['cells'] == []


def test_full_campaign_records_unmet_root_floor_without_admitting_training(tmp_path, monkeypatch):
    from collections import namedtuple
    runner, repo, preparation, prefix, cuda, destination, output, cells, preserved = campaign_fixture(tmp_path, monkeypatch)
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda path:
        usage(100 * 1024**3, 0, (19 if str(path) == '/' else 40) * 1024**3))
    result = runner.run(repo, preparation, prefix, cuda, destination, output)
    assert result['status'] == 'PASS_AUTHENTICATED_INPUT_RELOCATION'
    assert result['root_storage_health_floor_met'] is False
    assert result['training_resource_admitted'] is False
