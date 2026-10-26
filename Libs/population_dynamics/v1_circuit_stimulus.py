"""Generate source visual inputs bound to a constructed LGN/background population."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import yaml

from .provenance import freeze_json, sha256_file, versions
from .v1_configured_inputs import (compare_population, delivered_file, read_h5_events,
                                   read_input_inventory, validate_plan)
from .v1_lgn_source import read_input_construction_cells
from .v1_lgn_runtime import load_prepared_runtime
from .v1_lgn_spikes import (export_source_spikes, generate_source_spikes,
                            input_plan_identity, write_spike_trial)
from .v1_nest_input_replay import bound_json
from .v1_nest_time_semantics import V1_DURATION_MS, V1_RESOLUTION_MS

COMPLETE = 'CONSTRUCTED_VISUAL_INPUTS_COMPLETE_DEVELOPMENT'
REPRESENTATIVES = 'CONSTRUCTED_VISUAL_REPRESENTATIVES_COMPLETE_DEVELOPMENT'


def rate_config_without_noise(config: dict) -> dict:
    """Compare rate construction while separating two source sampler invocations.

    Keep every other field, including the source clock and trial count. Missing
    or non-integer seeds cannot become equivalent by removing their keys.
    """
    if not isinstance(config, dict):
        raise ValueError('V1_CONSTRUCTED_VISUAL_INTEGER_NOISE_SEEDS_REQUIRED')
    comparison = dict(config)
    for section in ('spike_generation', 'background'):
        settings = config.get(section)
        if not isinstance(settings, dict) or type(settings.get('starting_noise_seed')) is not int:
            raise ValueError('V1_CONSTRUCTED_VISUAL_INTEGER_NOISE_SEEDS_REQUIRED')
        comparison[section] = {key: value for key, value in settings.items()
                               if key != 'starting_noise_seed'}
    return comparison


def background_rates(settings: dict, duration_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """The declared development background uses the same pinned source sampler."""
    if (settings.get('kind') != 'source_constant_rate_sampler'
            or settings.get('rate_clock_step_ms') != V1_RESOLUTION_MS
            or type(settings.get('starting_noise_seed')) is not int
            or isinstance(settings.get('rate_hz'), bool)
            or not isinstance(settings.get('rate_hz'), (int, float))
            or not np.isfinite(settings['rate_hz']) or settings['rate_hz'] < 0
            or duration_ms != V1_DURATION_MS):
        raise ValueError('V1_CONSTRUCTED_BACKGROUND_SPECIFICATION_INVALID')
    clock = np.arange(int(duration_ms / V1_RESOLUTION_MS) + 1) * V1_RESOLUTION_MS / 1000.0
    return clock, np.full((1, len(clock)), settings['rate_hz'], dtype=np.float64)


def generate_inputs(rate_archive: Path, construction_archive: Path,
                    config_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='FAILED_CONSTRUCTED_VISUAL_INPUTS', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), versions=versions(),
        scientific_qualification=False, source_fidelity_qualified=False,
        production_episodes=0, coupled_neural_episodes=0,
        historical_background_rng_reproduced=False, training_seeds_used=[])
    try:
        config = yaml.safe_load(config_path.read_text())
        rate, delivery = bound_json(rate_archive, 'probe/probe.json')
        cells, _, binding = read_input_construction_cells(construction_archive)
        if (delivery.get('exit_code') != 0
                or rate.get('status') != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
                or rate.get('input_construction_binding') != binding
                or rate_config_without_noise(rate.get('config')) != rate_config_without_noise(config)
                or type(config['spike_generation'].get('trials')) is not int
                or config['spike_generation']['trials'] != 1
                or config['stimulus']['duration_s'] * 1000 != V1_DURATION_MS):
            raise ValueError('V1_CONSTRUCTED_VISUAL_RATE_BINDING_DIFFERS')
        coverage = config['unit_coverage']
        selected = cells if coverage == 'all_source_units' else cells.groupby('model_id', sort=False).head(1)
        if (coverage not in ('all_source_units', 'model_representatives')
                or selected.index.tolist() != rate['source_unit_ids']
                or selected['model_id'].tolist() != rate['source_models']):
            raise ValueError('V1_CONSTRUCTED_VISUAL_RATE_UNIT_BINDING_DIFFERS')
        report.update(config=config, config_sha256=sha256_file(config_path),
            rate_config=rate['config'],
            input_construction_binding=binding,
            rate_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
            prepared_runtime_sha256=sha256_file(rate_archive/'prepared/prepared.json'),
            source_clock=dict(duration_ms=V1_DURATION_MS, resolution_ms=V1_RESOLUTION_MS),
            complete_lgn_population=coverage == 'all_source_units')
        lgn = export_source_spikes(rate_archive, config_path, output/'lgn')
        if lgn.get('input_construction_binding') != binding:
            raise ValueError('V1_CONSTRUCTED_SPIKE_BINDING_LOST')
        lgn_trial = lgn['trials'][0]
        nodes = delivered_file(construction_archive, 'primary/glif/bkg_nodes.h5')
        types = delivered_file(construction_archive, 'primary/glif/bkg_node_types.csv')
        bkg_ids, _ = read_input_inventory(nodes, types, 'bkg')
        if len(bkg_ids) != 1:
            raise ValueError('V1_CONSTRUCTED_SINGLE_BACKGROUND_REQUIRED')
        functions, poisson = load_prepared_runtime(rate_archive/'prepared')
        clock, rates = background_rates(config['background'], V1_DURATION_MS)
        trains, seeds = generate_source_spikes(clock, rates, bkg_ids, 1,
            config['background']['starting_noise_seed'],
            functions.f_rate_to_spike_train, poisson.generate_inhomogenous_poisson)
        (output/'bkg').mkdir()
        np.savez_compressed(output/'bkg/rates.npz', time_s=clock, rates_hz=rates, source_ids=bkg_ids)
        bkg_trial = write_spike_trial(bkg_ids, trains[0], seeds, V1_DURATION_MS,
                                     output/'bkg/trial_0000', population_name='bkg')
        report['populations'] = {}
        for name, row in [('lgn', lgn_trial), ('bkg', bkg_trial)]:
            report['populations'][name] = dict(row, plan=name+'/'+row['plan'], sonata=name+'/'+row['sonata'])
        report.update(status=COMPLETE if report['complete_lgn_population'] else REPRESENTATIVES,
            exit_code=0, lgn_report_sha256=sha256_file(output/'lgn/report.json'),
            background_rng_policy='Separate source-sampler invocation with its declared starting seed; no per-unit independence claim.',
            lgn_noise_seed=config['spike_generation']['starting_noise_seed'],
            background_noise_seed=config['background']['starting_noise_seed'])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            artifacts={p.relative_to(output).as_posix(): sha256_file(p)
                       for p in output.rglob('*') if p.is_file()})
        freeze_json(output/'report.json', report)
    return report


def load_inputs(archive: Path, expected_binding: dict, *,
                require_complete: bool = True) -> tuple[dict, dict]:
    """Validate both SONATA files and full inventories before a native consumer."""
    report, delivery = bound_json(archive, 'visual_inputs/report.json')
    if (delivery.get('exit_code') != 0 or report.get('exit_code') != 0
            or report.get('status') not in (COMPLETE, REPRESENTATIVES)
            or report.get('input_construction_binding') != expected_binding
            or report.get('source_clock') != dict(duration_ms=V1_DURATION_MS, resolution_ms=V1_RESOLUTION_MS)
            or set(report.get('populations', {})) != {'lgn', 'bkg'}):
        raise ValueError('V1_CONSTRUCTED_INPUT_BUNDLE_BINDING_DIFFERS')
    full = report['status'] == COMPLETE
    if report.get('complete_lgn_population') is not full or (require_complete and not full):
        raise ValueError('V1_CONSTRUCTED_COMPLETE_VISUAL_POPULATION_REQUIRED')
    plans = []
    for name, expected_units in [('lgn', 17400 if full else 14), ('bkg', 1)]:
        row = report['populations'][name]
        paths = {}
        for key in ('plan', 'sonata'):
            relative = Path(row[key])
            if relative.parts[0] != name or len(relative.parts) != 2 or '..' in relative.parts:
                raise ValueError('V1_CONSTRUCTED_INPUT_ARTIFACT_PATH_INVALID')
            paths[key] = delivered_file(archive, 'visual_inputs/'+relative.as_posix())
            if sha256_file(paths[key]) != report['artifacts'][relative.as_posix()]:
                raise ValueError('V1_CONSTRUCTED_INPUT_ARTIFACT_CHANGED')
        with np.load(paths['plan'], allow_pickle=False) as source:
            plan = {key: source[key] for key in source.files}
        validate_plan(plan)
        if (len(plan['source_id']) != expected_units or row['source_units'] != expected_units
                or np.any(plan['population'] != 0)
                or (plan['source_event_time_ms'] > V1_DURATION_MS).any()
                or not np.array_equal(plan['source_event_unit'],
                                      np.repeat(np.arange(expected_units), plan['source_counts']))
                or input_plan_identity(plan) != row['input_identity_sha256']):
            raise ValueError('V1_CONSTRUCTED_INPUT_INVENTORY_OR_IDENTITY_DIFFERS')
        ids, times, metadata = read_h5_events(paths['sonata'], V1_DURATION_MS)
        if (metadata['source_population_labels'] != [name]
                or not compare_population(plan, 0, ids, times)['all_event_pairs_exact']):
            raise ValueError('V1_CONSTRUCTED_INPUT_SONATA_DIFFERS')
        plans.append(plan)
    # NumPy promotes mixed int64/uint64 concatenation to float64. Preserve
    # native per-population files, then use a checked integer join representation.
    for plan in plans:
        if np.any(plan['source_id'] > np.iinfo(np.int64).max):
            raise ValueError('V1_CONSTRUCTED_SOURCE_ID_EXCEEDS_NATIVE_INT64')
    merged_ids = [p['source_id'].astype(np.int64, copy=False) for p in plans]
    offset = len(plans[0]['source_id'])
    merged = dict(population=np.concatenate([np.zeros(offset, dtype=np.int64), np.ones(1, dtype=np.int64)]),
        source_id=np.concatenate(merged_ids),
        source_counts=np.concatenate([p['source_counts'] for p in plans]),
        source_event_unit=np.concatenate([plans[0]['source_event_unit'], plans[1]['source_event_unit']+offset]),
        source_event_time_ms=np.concatenate([p['source_event_time_ms'] for p in plans]))
    validate_plan(merged)
    merged['trains'] = np.split(merged['source_event_time_ms'], np.cumsum(merged['source_counts'])[:-1])
    report = dict(report, bundle_report_sha256=sha256_file(archive/'visual_inputs/report.json'),
                  input_identity_sha256=input_plan_identity(merged))
    return merged, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rate-archive', 'input-construction', 'config', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    report = generate_inputs(args.rate_archive, args.input_construction, args.config, args.output)
    print(report['status'])


if __name__ == '__main__':
    main()
