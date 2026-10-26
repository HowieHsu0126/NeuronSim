"""Software fixtures for disjoint PFC partitions; never scientific observations."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

_script = Path(__file__).resolve().parents[1]/'scripts/summarize_pfc_coupled_clock.py'
_spec = importlib.util.spec_from_file_location('pfc_clock_summary_test_target', _script)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def protocol():
    return dict(schema='pfc_coupled_clock_recurrent_gain_replicates_v1',
        independent_blocks_per_size=29, full_existing_blocks_per_size=32,
        analysis_partitions=['pilot3', 'remaining29', 'full32_descriptive'],
        newly_selected_independent_source_blocks=0, new_fine_source_episodes=348,
        coarse_source_episodes_reused=174, reused_pilot_fine_source_episodes=36,
        confirmation_execution_released=False, scientific_fidelity_qualified=False,
        cohorts={name: dict(source_rng_seeds=list(range(start+3, start+32)))
                 for name, start in [('reference_2560', 93001), ('expanded_100k', 95001)]},
        reused_pilot_readout_sha256='synthetic-fixture-reference')


def moments(values, endpoint='rate'):
    return dict(cohort='expanded_100k', metric=endpoint, blocks=len(values),
        value_mean=statistics.fmean(values), value_block_sd=statistics.stdev(values),
        value_min=min(values), value_max=max(values))


def test_registered_banks_are_complete_and_disjoint():
    pilot = summary.registered_blocks(None, {})
    remaining = summary.registered_blocks(protocol(),
        {'scope': 'development_recurrent_gain_coupled_clock_replicates_comparison'})
    for cohort in pilot:
        assert len(pilot[cohort]) == 3 and len(remaining[cohort]) == 29
        assert not pilot[cohort] & remaining[cohort]
        assert len(pilot[cohort] | remaining[cohort]) == 32


@pytest.mark.parametrize('change', ['overlap', 'duplicate', 'missing', 'qualification'])
def test_registry_rejects_exposed_or_incomplete_bank(change):
    config = protocol()
    seeds = config['cohorts']['expanded_100k']['source_rng_seeds']
    if change == 'overlap': seeds[0] = 95001
    elif change == 'duplicate': seeds[-1] = seeds[0]
    elif change == 'missing': seeds.pop()
    else: config['scientific_fidelity_qualified'] = True
    with pytest.raises(ValueError):
        summary.registered_blocks(config,
            {'scope': 'development_recurrent_gain_coupled_clock_replicates_comparison'})


def test_pooling_matches_all_individual_values_and_keeps_partitions():
    pilot = [-3., 2., 8.]
    remaining = [float(x) for x in range(10, 39)]
    rows = summary.partition_summaries([moments(remaining)], [moments(pilot)],
                                       ['cohort', 'metric'], ['value'])
    assert [row['partition'] for row in rows] == ['pilot3', 'remaining29', 'full32_descriptive']
    assert [int(row['blocks']) for row in rows] == [3, 29, 32]
    pooled = rows[-1]
    assert pooled['value_mean'] == pytest.approx(statistics.fmean(pilot+remaining))
    assert pooled['value_block_sd'] == pytest.approx(statistics.stdev(pilot+remaining))
    assert pooled['value_min'] == -3 and pooled['value_max'] == 38
    assert {k: v for k, v in rows[0].items() if k != 'partition'} == moments(pilot)


def test_constant_nonzero_block_means_allow_floating_point_roundoff():
    rows = summary.partition_summaries([moments([.1]*29)], [moments([.1]*3)],
                                       ['cohort', 'metric'], ['value'])
    assert rows[-1]['value_mean'] == pytest.approx(.1)
    assert rows[-1]['value_block_sd'] == pytest.approx(0., abs=1e-15)


@pytest.mark.parametrize('change', ['missing_endpoint', 'duplicate_endpoint', 'wrong_n',
                                   'nonfinite', 'negative_sd', 'mean_outside_support'])
def test_endpoint_pooling_refuses_invalid_or_mismatched_partitions(change):
    left, right = [moments([1., 2., 3.])], [moments(list(range(29)))]
    if change == 'missing_endpoint': right[0]['metric'] = 'other'
    elif change == 'duplicate_endpoint': right.append(copy.deepcopy(right[0]))
    elif change == 'wrong_n': right[0]['blocks'] = 28
    elif change == 'nonfinite': right[0]['value_mean'] = float('nan')
    elif change == 'negative_sd': right[0]['value_block_sd'] = -1.
    else: right[0]['value_mean'] = 100.
    with pytest.raises(ValueError):
        summary.partition_summaries(right, left, ['cohort', 'metric'], ['value'])


def block_rows(blocks):
    return [dict(cohort=cohort, source_rng_seed=seed, condition='cue', population='E')
            for cohort, seeds in blocks.items() for seed in sorted(seeds)]


@pytest.mark.parametrize('change', ['valid', 'missing', 'duplicate', 'pilot_overlap'])
def test_per_block_rows_are_retained_once_without_pseudoreplication(change):
    expected = summary.registered_blocks(protocol(),
        {'scope': 'development_recurrent_gain_coupled_clock_replicates_comparison'})
    first = block_rows(summary.registered_blocks(None, {}))
    rest = block_rows(expected)
    if change == 'missing': rest.pop()
    elif change == 'duplicate': rest.append(copy.deepcopy(rest[0]))
    elif change == 'pilot_overlap': rest[0]['source_rng_seed'] = 93001
    if change != 'valid':
        with pytest.raises(ValueError):
            summary.partition_block_rows(rest, first, expected, ['condition', 'population'])
    else:
        rows = summary.partition_block_rows(rest, first, expected, ['condition', 'population'])
        assert len(rows) == 64
        assert sum(row['partition'] == 'pilot3' for row in rows) == 6
        assert sum(row['partition'] == 'remaining29' for row in rows) == 58
        assert all(row['partition'] != 'full32_descriptive' for row in rows)


def delivered(root, files, **header):
    records = []
    for name, value in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value)+'\n').encode()
        path.write_bytes(data)
        records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (root/'delivery.json').write_text(json.dumps(dict(files=records, **header)))


def test_pilot_summary_requires_matching_readout_and_untampered_delivery(tmp_path):
    report = dict(status='COMPLETE_DESCRIPTIVE_COUPLED_CLOCK_SUMMARY', complete_source_episodes=54,
        fine_source_episodes=36, coarse_episodes_reused=18, original_independent_blocks_per_size=3,
        condition_pairing_rows_passed=54, input_sha256={'analysis/report.json': 'wrong-fixture-reference'},
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        confidence_intervals=False, p_values=False, cross_size_pairing=False, outputs={})
    delivered(tmp_path, {'summary/report.json': report}, status='COMPLETE', exit_code=0, job_id='fixture')
    with pytest.raises(ValueError, match='matching accepted pilot3'):
        summary.read_pilot_summary(tmp_path, protocol())
    (tmp_path/'summary/report.json').write_text('{}')
    with pytest.raises(ValueError, match='delivered pilot summary differs'):
        summary.read_pilot_summary(tmp_path, protocol())


def test_real_cli_rejects_synthetic_readout_before_creating_scientific_output(tmp_path):
    source = tmp_path/'synthetic_source'
    source.mkdir()
    report = dict(status='COMPLETE_SYNTHETIC_RECURRENT_GAIN_READOUT_VALIDATION')
    delivered(source, {'report.json': {'exit_code': 0}, 'analysis/report.json': report},
              verified=True, status=report['status'], job_id='fixture')
    output = tmp_path/'must_not_exist'
    command = [sys.executable, '-B', str(Path(summary.__file__)), '--source', str(source), '--output', str(output)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode != 0
    assert 'complete descriptive registered source readout required' in run.stderr
    assert not output.exists()


def test_remaining_mode_requires_both_protocol_and_pilot_summary(tmp_path):
    with pytest.raises(ValueError, match='must be supplied together'):
        summary.summarize(tmp_path/'absent', tmp_path/'output', protocol_path=tmp_path/'protocol.yaml')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('failure', ['partial_bank', 'changed_protocol', 'unaccepted_software'])
def test_launcher_preserves_failed_admission_without_starting_summary(tmp_path, failure):
    repo = tmp_path/'repo'
    protocol_name = 'Libs/configs/neural_population_pfc_recurrent_gain_coupled_clock_replicates.yaml'
    target = repo/protocol_name
    target.parent.mkdir(parents=True)
    target.write_text('synthetic launcher fixture\n')
    validation, source, pilot = (tmp_path/name for name in ('validation', 'source', 'pilot'))
    for path in (validation, source, pilot): path.mkdir()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    vr = {'implementation_sha256': {protocol_name: digest if failure != 'changed_protocol' else 'different'}}
    delivered(validation, {'report.json': vr}, status='FAILED' if failure == 'unaccepted_software'
              else 'PASS_SOFTWARE_AND_ACCEPTED_PILOT_INTAKE', exit_code=0)
    delivered(source, {'analysis/report.json': {'complete_source_episodes_consumed': 54, 'blocks': [0]*6}},
              status='SYNTHETIC_LAUNCHER_FIXTURE')
    (pilot/'delivery.json').write_text(json.dumps({'files': []}))
    script = _script.with_name('run_pfc_coupled_clock_summary.sbatch')
    env = dict(os.environ, PFC_PYTHON=sys.executable, SLURM_JOB_ID='isolated_fixture')
    run = subprocess.run(['bash', str(script), str(repo), str(source), str(pilot), str(validation)],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 1
    output = repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/kernel_parameterization/recurrent_gain/coupled_clock_replicates/summary/run_isolated_fixture'
    receipt = json.loads((output/'execution.json').read_text())
    expected = {'partial_bank': 'complete remaining29 reader output required',
                'changed_protocol': 'registered source protocol changed after validation',
                'unaccepted_software': 'accepted summary software required'}
    assert expected[failure] in receipt['error']
    assert receipt['status'] == 'FAILED'
    assert 'command' not in receipt
    assert not (output/'summary').exists()
