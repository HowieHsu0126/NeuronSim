"""Relocate authenticated LDNS input copies to compute-node-local storage.

The original cell keeps its receipts and input path. Only a verified duplicate
input tree is removed after the original path links to the verified destination.
Training and source simulation are never started by this command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, record):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verified_files(root, rows):
    expected = {}
    for row in rows:
        name = row['path']
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or str(relative) != name
                or name in expected or type(row['bytes']) is not int or row['bytes'] < 0):
            raise ValueError('LDNS relocation requires unique safe manifest paths and byte lengths')
        expected[name] = row
    paths = list(root.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('LDNS relocation rejects links inside an input tree')
    actual = {str(path.relative_to(root)) for path in paths if path.is_file()}
    if actual != set(expected):
        raise ValueError('LDNS relocation input tree differs from its complete manifest')
    for name, row in expected.items():
        path = root / name
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('LDNS relocation input file changed: ' + name)
    return sum(row['bytes'] for row in expected.values())


def relocate(cell, destination, rows):
    """Copy and verify before switching the original path; preserve all bytes."""
    cell, destination = Path(cell), Path(destination).resolve()
    source = cell / 'inputs'
    if source.is_symlink():
        if source.resolve() != destination:
            raise ValueError('LDNS existing input link has another destination')
        size = verified_files(source, rows)
        return dict(status='ALREADY_RELOCATED_VERIFIED', source=str(source),
                    destination=str(destination), files=len(rows), bytes=size)
    if destination.exists():
        raise FileExistsError('LDNS relocation destination exists; resolve the previous outcome')
    if destination == source.resolve() or source.resolve() in destination.parents:
        raise ValueError('LDNS relocation destination must be separate from its source')
    backup = cell / 'inputs.relocation_backup'
    link = cell / 'inputs.relocation_link'
    if backup.exists() or link.exists() or link.is_symlink():
        raise FileExistsError('LDNS incomplete path switch must be reconciled before retry')
    size = verified_files(source, rows)
    if shutil.disk_usage(destination.parent).free < size + 20 * 1024**3:
        raise RuntimeError('LDNS destination lacks copy space above the existing 20 GiB reserve')
    destination.mkdir()
    for row in rows:
        target = destination / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row['path'], target)
    verified_files(destination, rows)
    # A source producer must not have changed files while the copy was made.
    verified_files(source, rows)
    allocated = sum((source / row['path']).stat().st_blocks * 512 for row in rows)
    link.symlink_to(destination, target_is_directory=True)
    source.rename(backup)
    try:
        link.rename(source)
    except BaseException:
        backup.rename(source)
        link.unlink(missing_ok=True)
        raise
    # Both the original path and the independent destination have authenticated
    # bytes before this task removes its now-redundant local copy.
    verified_files(source, rows)
    shutil.rmtree(backup)
    return dict(status='RELOCATED_VERIFIED_INPUT_COPY', source=str(source),
                destination=str(destination), files=len(rows), bytes=size,
                original_allocated_bytes=allocated, original_path_is_link=True,
                original_receipts_unchanged=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run(repo, preparation, failed_prefix, cuda_cell, destination_root, output):
    if (os.environ.get('SLURMD_NODENAME') != 'gpu03'
            or os.environ.get('SLURM_JOB_PARTITION') != 'gpu1'
            or not os.environ.get('SLURM_JOB_GPUS')):
        raise ValueError('LDNS relocation requires an allocated GPU03/gpu1 job')
    repo, preparation, cuda_cell, destination_root, output = map(
        Path, (repo, preparation, cuda_cell, destination_root, output))
    if Path('/data') not in destination_root.resolve().parents:
        raise ValueError('LDNS relocation destination must use GPU03 /data storage')
    if destination_root.exists() or output.exists():
        raise FileExistsError('LDNS relocation state exists; inspect before repeating')
    if destination_root.parent.stat().st_dev == Path('/').stat().st_dev:
        raise ValueError('LDNS relocation destination must be a separate filesystem')
    output.mkdir(parents=True)
    record = dict(status='VALIDATING_LDNS_INPUT_RELOCATION', job_id=os.environ['SLURM_JOB_ID'],
                  recorded_utc=datetime.now(timezone.utc).isoformat(), cells=[],
                  training_started=False, source_simulation_started=False,
                  scientific_protocol_changed=False, destination_root=str(destination_root))
    path = output / 'relocation.json'
    write(path, record)
    try:
        telemetry = subprocess.check_output(['vmstat', '1', '4'], text=True)
        lines = telemetry.strip().splitlines()
        headings = lines[1].split()
        values = [dict(zip(headings, line.split())) for line in lines[2:]]
        if (all(float(row['wa']) >= 25 for row in values[-3:])
                or all(float(row['si']) > 0 or float(row['so']) > 0 for row in values[-3:])):
            raise RuntimeError('LDNS relocation waits for stable I/O and swap telemetry')
        record['vmstat_before'] = telemetry
        record['root_free_before_bytes'] = shutil.disk_usage('/').free
        resource = module(preparation / 'Libs/population_dynamics/resource_probe.py',
                          'ldns_relocation_resource_probe')
        snapshot = resource.snapshot()
        if snapshot['d_state_count'] >= 15:
            raise RuntimeError('LDNS relocation waits for uninterruptible I/O tasks to clear')
        record['resource_snapshot_before'] = snapshot
        stager = module(preparation / 'Libs/scripts/run_population_trajectory_campaign.py',
                        'ldns_retained_stager')
        cohort = repo / stager.COHORT
        receipt = json.loads((cohort / 'delivery.json').read_text())
        receipt_sha = digest(cohort / 'delivery.json')
        authoritative = stager.manifest_rows(receipt)
        cells = [Path(str(failed_prefix) + '-' + str(seed)) for seed in (41, 42, 43)] + [cuda_cell]
        plans = []
        for index, cell in enumerate(cells):
            if any((cell / name).exists() for name in ('training', 'forecasts', 'archive_complete')):
                raise ValueError('LDNS input cell has training or generated results')
            if index < 3:
                health = json.loads((cell / 'phases/train/preflight.json').read_text())
                if health['status'] != 'BLOCKED' or health['reasons'] != ['ROOT_STORAGE_CRITICAL']:
                    raise ValueError('LDNS relocation requires the original pretraining storage failure')
                if digest(cell / 'preparation.json') != digest(preparation / 'preparation.json'):
                    raise ValueError('LDNS relocation original preparation changed')
            else:
                gate = repo / 'Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012'
                gate_receipt = json.loads((gate / 'delivery.json').read_text())
                if gate_receipt['status'] != 'PASS':
                    raise ValueError('LDNS relocation requires accepted CUDA qualification')
                for row in gate_receipt['files']:
                    if digest(cell / 'results' / row['path']) != row['sha256']:
                        raise ValueError('LDNS original CUDA result differs from accepted delivery')
            inputs = json.loads((cell / 'inputs.json').read_text())
            declared = stager.manifest_rows(inputs)
            if (inputs['delivery_sha256'] != receipt_sha
                    or digest(cell / 'prerequisites/cohort_delivery.json') != receipt_sha
                    or len(declared) != 436 or any(row != authoritative[name] for name, row in declared.items())):
                raise ValueError('LDNS relocation requires complete original cohort bindings')
            plans.append((cell, destination_root / cell.name, list(declared.values())))
            verified_files(cell / 'inputs', plans[-1][2])
        record['validated_cells'] = [str(cell) for cell in cells]
        record['status'] = 'VALIDATED_ALL_INPUT_CELLS'
        write(path, record)
        destination_root.mkdir()
        for cell, destination, rows in plans:
            result = relocate(cell, destination, rows)
            record['cells'].append(result)
            write(path, record)
        record.update(status='PASS_AUTHENTICATED_INPUT_RELOCATION',
                      root_free_after_bytes=shutil.disk_usage('/').free,
                      destination_free_after_bytes=shutil.disk_usage(destination_root).free,
                      relocated_bytes=sum(row['bytes'] for row in record['cells']),
                      root_storage_health_floor_met=shutil.disk_usage('/').free >= 20 * 1024**3,
                      training_resource_admitted=False,
                      recorded_utc=datetime.now(timezone.utc).isoformat())
        write(path, record)
    except BaseException as error:
        record.update(status='INCOMPLETE_LDNS_INPUT_RELOCATION',
                      error=dict(type=type(error).__name__, message=str(error)))
        write(path, record)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'preparation', 'failed-prefix', 'cuda-cell', 'destination-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.preparation, args.failed_prefix, args.cuda_cell,
                         args.destination_root, args.output)))


if __name__ == '__main__':
    main()
