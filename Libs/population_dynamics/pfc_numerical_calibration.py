"""Prepare a fixed, development-only PFC numerical calibration cohort."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import yaml
from .pfc_scale_analysis import _source_registration, CONDITIONS
from .pfc_delay_analysis import require, write_json
from .provenance import sha256_file

SCHEMA = 'pfc_numerical_calibration_v1'


def prepare(protocol_path, output, project_root):
    protocol_path, output, project_root = map(Path, (protocol_path, output, project_root))
    protocol = yaml.safe_load(protocol_path.read_text())
    require(protocol['schema'] == SCHEMA and protocol['scope'] == 'development_calibration'
            and protocol['purpose'] == 'resolve_sampling_precision_of_native_clock_comparison',
            'registered development numerical calibration required')
    seeds = protocol['joint_rng_seeds']
    require(isinstance(seeds, list) and len(seeds) == protocol['joint_rng_blocks']
            and len(seeds) >= 3 and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
            and seeds == sorted(set(seeds)), 'complete unique ordered calibration seeds required')
    forbidden = set(protocol['previously_inspected_joint_rng_seeds'])
    reserve = protocol['confirmation_seed_reservation']
    require(set(seeds).isdisjoint(forbidden) and set(seeds).isdisjoint(reserve)
            and len(reserve) == len(set(reserve)) and set(reserve).isdisjoint(forbidden),
            'calibration, inspected and reserved confirmation seeds must be disjoint')
    require(protocol['independent_unit'] == 'joint_initial_background_rng_block'
            and protocol['coupled_circuit_instances'] == 1
            and protocol['training_seeds'] == [41, 42, 43]
            and protocol['confirmation_execution_released'] is False
            and protocol['source_scientific_qualification_granted'] is False,
            'source/calibration/training authority differs')
    sources = protocol['source_configs']
    require(len(sources) == 3 and {row['dt_ms'] for row in sources} == {.05, .025, .0125},
            'all three source clocks required')
    batch_size = protocol['execution']['cases_per_allocation']
    require(type(batch_size) is int and batch_size > 0, 'positive case packing size required')
    configs = []
    for source in sources:
        relative = Path(source['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'repository-relative source configuration required')
        path = project_root/relative
        config, _, inspected, dimensions = _source_registration(path)
        require(dimensions['coupled_core_units'] == 100000 and config['dt_ms'] == source['dt_ms']
                and set(inspected) <= forbidden and 'source_replication' not in config,
                'unchanged inspected100k source configuration required')
        configs.append((source, path, config))
    canonical = {k: v for k, v in configs[0][2].items() if k != 'dt_ms'}
    require(all({k: v for k, v in config.items() if k != 'dt_ms'} == canonical
                for _, _, config in configs), 'only clock may differ among source configurations')
    output.mkdir(parents=True, exist_ok=False)
    clocks = []
    for source, path, original in configs:
        config = copy.deepcopy(original)
        config['source_rng_seed'] = seeds[0]
        config['source_replication'] = dict(purpose='numerical_calibration',
            independent_unit=protocol['independent_unit'], joint_rng_seeds=seeds)
        config['cases'] = [dict(case_id=f'pfc_{seed}_{condition}', source_rng_seed=seed,
            stimulus_strength_nA=strength, distractor_strength_nA=0.)
            for seed in seeds for condition, strength in CONDITIONS.items()]
        tag = str(source['dt_ms']).replace('.', 'p')+'ms'
        destination = output/('source_dt_'+tag+'.yaml')
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
        _, registered, emitted_seeds, _ = _source_registration(destination)
        require(emitted_seeds == seeds and len(registered) == len(seeds)*3,
                'emitted calibration registration differs')
        case_ids = list(registered)
        batches = []
        for first in range(0, len(case_ids), batch_size):
            selected = case_ids[first:first+batch_size]
            name = 'source_dt_'+tag+'_cases_'+str(first).zfill(3)+'.json'
            write_json(output/name, selected)
            batches.append(dict(path=name, case_ids=selected))
        clocks.append(dict(dt_ms=source['dt_ms'], config=destination.name,
            config_sha256=sha256_file(destination), reference_config=str(path),
            reference_config_sha256=sha256_file(path), expected_cases=len(registered), batches=batches))
    report = dict(status='PREPARED_NUMERICAL_CALIBRATION', protocol_sha256=sha256_file(protocol_path),
        joint_rng_seeds=seeds, independent_joint_blocks=len(seeds), coupled_circuit_instances=1,
        clocks=clocks, expected_complete_source_episodes=3*3*len(seeds), production_episodes=0,
        new_source_simulations=0, confidence_intervals=False, p_values=False, q_values=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        source_marginal_random_law_changed=False, original_numerical_engine_changed=False,
        numerical_margins_frozen=protocol['numerical_margins_frozen'],
        implementation_sha256=sha256_file(Path(__file__)))
    write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    prepare(args.protocol, args.output, args.project_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
