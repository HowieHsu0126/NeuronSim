"""Bind missing PFC fine clocks to every block in the completed coarse bank.

Run on a compute node. This constructs a development protocol from existing
receipts; it neither reads response metrics nor executes source simulations.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(project_root, output):
    pilot_path = project_root / 'Libs/configs/neural_population_pfc_recurrent_gain_coupled_clock.yaml'
    expected_pilot = 'd126f8287def6a71a592a6b75ba0c30ef94f5e70af37ec9b24bd8442d3606ad1'
    assert digest(pilot_path) == expected_pilot, 'FROZEN_PILOT_PROTOCOL_CHANGED'
    pilot = yaml.safe_load(pilot_path.read_text())
    protocol = copy.deepcopy(pilot)
    protocol.update(
        schema='pfc_coupled_clock_recurrent_gain_replicates_v1',
        scope='development_coupled_clock_recurrent_gain_replicates',
        purpose='Test clock sensitivity of the complete existing recurrent-gain response bank, retaining all blocks and controls',
        created_utc=datetime.now(timezone.utc).isoformat(),
        owner_thread_id='01a0f19d-2b54-7440-a411-8e3c7bd4d845',
        pilot_protocol=str(pilot_path.relative_to(project_root)),
        pilot_protocol_sha256=expected_pilot,
        reused_pilot_readout='Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/kernel_parameterization/recurrent_gain/coupled_clock/readouts/run_597649/analysis/report.json',
        reused_pilot_readout_sha256='612afbb6402c8e75d0b66b3f8fd96914a3c12d48acf3bb2c55596ed013fd0e68',
        reused_pilot_fine_source_episodes=36,
        coarse_source_episodes_reused=174,
        new_fine_source_episodes=348,
        independent_blocks_per_size=29,
        full_existing_blocks_per_size=32,
        newly_selected_independent_source_blocks=0,
        analysis_partitions=['pilot3', 'remaining29', 'full32_descriptive'],
        p_values=False,
        claim_scope='Descriptive Jpos2.3 weak-cue, omission and strong-cue clock sensitivity in every previously executed coarse block. No fine-clock J1.6-J2.3 paired interaction, activation threshold, bistability, prevalence, numerical equivalence, parameter-domain qualification or production claim.',
        source_selection='Every remaining registered source block in order; no selection by response or outcome',
        refinement_seed_interpretation='The original block-index mapping is continued. Auxiliary RNG seeds are not new independent units. The expanded cohort reuses the default-gain calibration refinement mapping; no cross-gain claim is made here.',
        preparation_implementation_sha256=digest(Path(__file__)),
    )
    gain_root = Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/kernel_parameterization/recurrent_gain')
    for cohort, source_cohort, job, first_seed, first_aux in [
            ('reference_2560', 'reference_candidate', 597040, 93004, 98717),
            ('expanded_100k', 'expanded_candidate', 597041, 95004, 95417)]:
        bank = protocol['cohorts'][cohort]
        bank['source_rng_seeds'] = list(range(first_seed, first_seed+29))
        bank['blocks'] = []
        for index, seed in enumerate(bank['source_rng_seeds']):
            references = {}
            for condition in protocol['conditions']:
                relative = gain_root/'replicates'/source_cohort/'source_cases'/f'run_{job}_pfc_{seed}_{condition}'
                archive = project_root/relative
                delivery = json.loads((archive/'delivery.json').read_text())
                assert delivery['status'] == 'ARCHIVED_PFC_SOURCE_EXECUTION'
                assert delivery['result_status'] == 'PFC_SOURCE_EXECUTION_COMPLETE' and delivery['source_execution_complete'] is True
                files = {row['path']: row for row in delivery['files']}
                report_path = archive/'source_run/report.json'
                assert digest(report_path) == files['source_run/report.json']['sha256']
                report = json.loads(report_path.read_text())
                assert report['case_id'] == f'pfc_{seed}_{condition}' and report['source_execution_complete'] is True
                assert report['seeds'] == {'source_rng_seed': seed}
                assert report['resolved_config']['dt_ms'] == .05 and report['resolved_config']['duration_ms'] == 5000.
                references[condition] = dict(archive=str(relative), delivery_sha256=digest(archive/'delivery.json'),
                                             report_sha256=files['source_run/report.json']['sha256'])
            bank['blocks'].append(dict(source_rng_seed=seed,
                refinement_seeds=[first_aux+2*index, first_aux+2*index+1], coarse_references=references))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        yaml.safe_dump(protocol, stream, sort_keys=False)
    print(json.dumps(dict(status='PREPARED_COMPLETE_EXISTING_BLOCK_CLOCK_PROTOCOL', config=str(output),
        sha256=digest(output), coarse_cases_bound=174, missing_fine_cases=348,
        source_simulations=0, production_episodes=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.project_root, args.output)


if __name__ == '__main__':
    main()
