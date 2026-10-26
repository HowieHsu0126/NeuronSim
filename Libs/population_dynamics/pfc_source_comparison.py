"""Compare a source execution backend to all registered native PFC cases."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .provenance import freeze_json, sha256_file

PHYSICAL_FIELDS = (
    'excitatory_neurons', 'inhibitory_neurons', 'weight_scaling_factor', 'dt_ms',
    'duration_ms', 'stimulus_center_deg', 'stimulus_width_deg', 'stimulus_start_ms',
    'stimulus_duration_ms', 'distractor_center_deg', 'distractor_width_deg',
    'distractor_start_ms', 'distractor_duration_ms', 'state_monitored_per_population',
    'excitatory_preference_bins', 'readout_bin_ms')


def compare(candidate: Path | None, references: list[Path], design_path: Path, output: Path, *,
            candidate_kind: str = 'brian2', config_path: Path | None = None,
            source_runs: list[Path] | None = None, factorization_proof: Path | None = None):
    if output.exists():
        raise FileExistsError(output)
    if candidate_kind in ('streamed_numpy', 'streamed_cuda'):
        return _compare_streamed(candidate, references, design_path, output, config_path, source_runs,
                                 factorization_proof, candidate_kind=candidate_kind)
    if (candidate_kind != 'brian2' or candidate is None or config_path is not None or source_runs is not None
            or factorization_proof is not None):
        raise ValueError('PFC_SOURCE_COMPARISON_CANDIDATE_KIND_OR_ARGUMENTS_INVALID')
    design = json.loads(design_path.read_text())
    cr = json.loads((candidate / 'report.json').read_text())
    if (cr['status'] != 'NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS'
            or len(cr['cases']) != design['cases']
            or cr['resolved_config']['duration_ms'] != design['duration_ms']
            or sorted({c['source_rng_seed'] for c in cr['cases']}) != design['source_rng_blocks']):
        raise ValueError('PFC_CANDIDATE_COVERAGE_OR_STATUS_MISMATCH')
    # Numerical equivalence is a tighter question than scientific fidelity.
    # These source-execution tolerances are inherited unchanged from the
    # first registered comparison, not chosen from the resulting errors.
    if (design['relative_tolerance'] != 0.0
            or design['terminal_voltage_mV_max_absolute_error'] != 1e-10):
        raise ValueError('PFC_SOURCE_COMPARISON_TOLERANCE_CHANGED')
    index, bindings = {}, []
    for path in references:
        report = json.loads((path / 'report.json').read_text())
        if report['status'] != 'NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS':
            raise ValueError('PFC_REFERENCE_NOT_COMPLETE')
        for key in ('source_revision', 'source_entry_sha256', 'brian2_version', 'numpy_version'):
            if report[key] != cr[key]:
                raise ValueError('PFC_REFERENCE_SOURCE_MISMATCH:' + key)
        for key in PHYSICAL_FIELDS:
            if report['resolved_config'][key] != cr['resolved_config'][key]:
                raise ValueError('PFC_REFERENCE_PROTOCOL_MISMATCH:' + key)
        for case in report['cases']:
            if case['case_id'] in index:
                raise ValueError('PFC_DUPLICATE_REFERENCE_CASE')
            index[case['case_id']] = path, case
        bindings.append(dict(path=str(path.resolve()), report_sha256=sha256_file(path/'report.json')))
    if set(index) != {c['case_id'] for c in cr['cases']}:
        raise ValueError('PFC_REFERENCE_CASE_COVERAGE_MISMATCH')
    rows, total, maximum = [], 0, 0.0
    for case in cr['cases']:
        path, other = index[case['case_id']]
        for key in ('source_rng_seed', 'stimulus_strength_nA', 'distractor_strength_nA',
                    'core_units', 'native_events'):
            if case[key] != other[key]:
                raise ValueError('PFC_CASE_MISMATCH:' + case['case_id'] + ':' + key)
        a, b = candidate/case['raw_file'], path/other['raw_file']
        if sha256_file(a) != case['raw_sha256'] or sha256_file(b) != other['raw_sha256']:
            raise ValueError('PFC_NATIVE_ARTIFACT_CHANGED')
        compared, delta = 0, 0.0
        with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise ValueError('PFC_NATIVE_FIELDS_CHANGED')
            for name in left.files:
                x, y = left[name], right[name]
                if (x.shape != y.shape or x.dtype != y.dtype
                        or not np.isfinite(x).all() or not np.isfinite(y).all()):
                    raise ValueError('PFC_NATIVE_ARRAY_INVALID:' + name)
                if name == 'terminal_voltage_mV':
                    delta = float(np.max(np.abs(x-y)))
                    np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-10)
                else:
                    np.testing.assert_array_equal(x, y)
                compared += x.size
        total += compared
        maximum = max(maximum, delta)
        rows.append(dict(case_id=case['case_id'], source_rng_seed=case['source_rng_seed'],
            stimulus_strength_nA=case['stimulus_strength_nA'], values=compared,
            spike_arrays_exact=True, max_voltage_mV_difference=delta,
            native_seconds=other['elapsed_seconds'], candidate_seconds=case['elapsed_seconds']))
    result = dict(status='PASS_SOURCE_UNIFORM_PROJECTION_EQUIVALENCE', cases=rows,
        array_values=total, max_voltage_mV_difference=maximum, reference_bindings=bindings,
        candidate_report_sha256=sha256_file(candidate/'report.json'),
        design_sha256=sha256_file(design_path),
        comparison_implementation_sha256=sha256_file(Path(__file__)),
        source_revision=cr['source_revision'], source_entry_sha256=cr['source_entry_sha256'],
        execution_implementation_sha256=cr['execution_implementation_sha256'],
        full_scale_qualified=False, scientific_fidelity_qualified=False,
        cost_interpretation='same source outputs; separate executions, not a matched scientific speedup')
    freeze_json(output, result)
    return result



def _array_diagnostics(left, right):
    """Keep all fixed-tolerance failures, including different event counts."""
    rows = []
    for name in sorted(left):
        x, y = left[name], right[name]
        compatible = x.shape == y.shape and x.dtype == y.dtype
        voltage = name == 'terminal_voltage_mV'
        delta = float(np.max(np.abs(x-y), initial=0.)) if compatible and voltage else None
        passed = compatible and (np.allclose(x, y, rtol=0., atol=1e-10) if voltage
                                 else np.array_equal(x, y))
        rows.append(dict(array=name, passed=bool(passed), candidate_shape=list(x.shape),
            reference_shape=list(y.shape), candidate_dtype=str(x.dtype), reference_dtype=str(y.dtype),
            candidate_values=int(x.size), reference_values=int(y.size),
            values_compared=int(x.size) if compatible else 0,
            unequal_values=int(np.count_nonzero(x != y)) if compatible else None,
            max_voltage_mV_difference=delta, atol=1e-10 if voltage else 0., rtol=0.))
    return rows


REFERENCE_CLOCK_SCOPE = 'native_reference_clock_correspondence'


def _full_scale_requirements(source, candidate_backend):
    return dict(status=('REGISTERED_BEFORE_ARCHIVED_SOURCE_COMPARISON' if candidate_backend == 'cuda'
                            else 'REGISTERED_BEFORE_SOURCE_EXECUTION'), cases=9, units=100000,
        excitatory_neurons=80000, inhibitory_neurons=20000, weight_scaling_factor=.0256,
        dt_ms=.05, duration_ms=5000., source_rng_blocks=[94001,94002,94003],
        source_revision=source.SOURCE_REVISION, source_entry_sha256=source.SOURCE_ENTRY_SHA256,
        reference_synapse_execution='factorized_uniform_projections',
        relative_tolerance=0., terminal_voltage_mV_max_absolute_error=1e-10,
        required_event_and_integer_arrays='exact', candidate_backend=candidate_backend,
        all_case_numerical_failures_retained=True, full_scale_qualified=False,
        scientific_fidelity_qualified=False, production_episodes=0)


def _native_case_projection(source, config, case_ids):
    canonical_ids = [f'pfc_{seed}_{condition}' for seed in (94001,94002,94003)
                     for condition in ('cue','omission','strong_cue')]
    source.require([case['case_id'] for case in config['cases']] == canonical_ids,
                   'reference-clock canonical nine-case identity changed')
    source.require(isinstance(case_ids,list) and 1 <= len(case_ids) <= len(canonical_ids)
                   and all(isinstance(case,str) for case in case_ids)
                   and len(set(case_ids)) == len(case_ids) and set(case_ids) <= set(canonical_ids)
                   and case_ids == [case for case in canonical_ids if case in case_ids],
                   'explicit ordered canonical reference case projection required')
    projected = copy.deepcopy(config)
    projected['cases'] = [case for case in projected['cases'] if case['case_id'] in case_ids]
    return projected


def _reference_clock_requirements(source, config, case_ids):
    projected = _native_case_projection(source, config, case_ids)
    expected = _full_scale_requirements(source, 'numpy')
    expected.update(scope=REFERENCE_CLOCK_SCOPE, status='REGISTERED_BEFORE_NATIVE_REFERENCE_EXECUTION',
        cases=len(case_ids), dt_ms=.0125,
        source_rng_blocks=sorted({case['source_rng_seed'] for case in projected['cases']}),
        case_ids=case_ids, canonical_case_ids=[case['case_id'] for case in config['cases']],
        canonical_registered_cases=9, native_config_file='native_config.yaml',
        native_configuration_projection='cases_only', source_executions_preexist=True,
        native_reference_preexists=False, working_memory_qualified=False,
        fidelity_at_requested_scale_qualified=False)
    return expected


def _sha256_value(value):
    return isinstance(value,str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def prepare_reference(config_path: Path, source_runs: list[Path], factorization_proof: Path,
                      output: Path, case_ids: list[str]) -> dict:
    """Preregister existing source cases and a cases-only fresh native configuration."""
    from . import pfc_scale_analysis as source

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config, registered, _, dimensions = source._source_registration(config_path)
    config_sha = sha256_file(config_path)
    design = _reference_clock_requirements(source, config, case_ids)
    design.update(source_config_sha256=config_sha,
        reference_runtime=dict(brian2_version='2.5.4',numpy_version='1.26.4',scipy_version='1.12.0'),
        candidate_numpy_version='2.4.2',
        reference_execution_implementation_sha256=sha256_file(
            Path(factorization_proof)/'Libs/population_dynamics/pfc_factorized_source.py'),
        reference_factorization_parity_sha256=sha256_file(Path(factorization_proof)/'source_comparison.json'))
    headers = {}
    source.require(len(source_runs) == len(case_ids), 'complete preregistered source archive coverage required')
    for local in source_runs:
        status, header = source._source_header(local, config, registered, config_sha)
        source.require(header is not None and status['status'] == 'COMPLETE',
                       'completed preexisting reference-clock source required')
        case_id = header['report']['case_id']
        source.require(case_id in case_ids and case_id not in headers,
                       'duplicate or unregistered preexisting reference-clock case')
        report = header['report']
        source.require(report.get('backend') == 'numpy'
                       and report.get('driver',{}).get('numpy_version') == '2.4.2',
                       'preexisting reference-clock candidate runtime changed')
        source._source_execution_metadata(header,config,registered,dimensions,config_sha)
        headers[case_id] = header
    source.require(set(headers) == set(case_ids), 'complete preregistered source archive coverage required')
    design['preexisting_source_bindings'] = [dict(case_id=case_id,
        report_sha256=headers[case_id]['binding']['report_sha256'],
        delivery_sha256=headers[case_id]['binding']['delivery_sha256']) for case_id in case_ids]
    native = _native_case_projection(source, config, case_ids)
    native_bytes = yaml.safe_dump(native, sort_keys=False).encode('utf-8')
    design['native_config_sha256'] = hashlib.sha256(native_bytes).hexdigest()
    _streamed_design_contract(source, design, config, config_sha, factorization_proof)
    output.mkdir(parents=True)
    (output/'native_config.yaml').write_bytes(native_bytes)
    freeze_json(output/'design.json', design)
    return design


def _streamed_design_contract(source, design, config, config_sha, factorization_proof, *,
                              candidate_backend='numpy'):
    """Keep the 2560-unit contract separate from the frozen 100k comparison."""
    require = source.require
    require(candidate_backend in ('numpy', 'cuda'), 'unsupported streamed candidate backend')
    reference_clock = design.get('scope') == REFERENCE_CLOCK_SCOPE
    if design.get('scope') not in ('native_full_scale_source_equivalence', REFERENCE_CLOCK_SCOPE):
        require(candidate_backend == 'numpy', 'CUDA source comparison requires the full-scale design')
        require(factorization_proof is None, 'factorization proof requires the explicit full-scale design')
        native_config = dict(config)
        del native_config['synapse_execution']
        source.config_contract(design, native_config, config)
        return dict(full_scale_design=False, reference_synapse_execution='native_source',
                    reference_execution_implementation_sha256=source.SOURCE_ENTRY_SHA256)

    expected = _full_scale_requirements(source, candidate_backend)
    expected_ids = [f'pfc_{seed}_{condition}' for seed in expected['source_rng_blocks']
                    for condition in ('cue','omission','strong_cue')]
    if reference_clock:
        require(candidate_backend == 'numpy', 'reference-clock correspondence requires NumPy source')
        expected = _reference_clock_requirements(source, config, design.get('case_ids'))
        expected_ids = expected['case_ids']
        bindings = design.get('preexisting_source_bindings')
        require(_sha256_value(design.get('native_config_sha256'))
                and isinstance(bindings,list) and len(bindings) == len(expected_ids)
                and [row.get('case_id') for row in bindings if isinstance(row,dict)] == expected_ids
                and all(set(row) == {'case_id','report_sha256','delivery_sha256'}
                        and _sha256_value(row['report_sha256']) and _sha256_value(row['delivery_sha256'])
                        for row in bindings), 'reference-clock preregistration binding missing or changed')
    require(all(design.get(key) == value for key,value in expected.items())
            and design.get('full_scale_qualified') is False
            and design.get('scientific_fidelity_qualified') is False
            and design.get('all_case_numerical_failures_retained') is True
            and design.get('case_ids') == expected_ids
            and [case['case_id'] for case in config['cases']] ==
                (expected['canonical_case_ids'] if reference_clock else expected_ids),
            'registered full-scale source comparison design required')
    require(design.get('source_config_sha256') == config_sha
            and all(config.get(key) == expected[key] for key in (
                'excitatory_neurons','inhibitory_neurons','weight_scaling_factor','dt_ms','duration_ms'))
            and config.get('synapse_execution') == expected['reference_synapse_execution'],
            'full-scale comparison source config bytes or scale changed')
    require(design.get('reference_runtime') == dict(
                brian2_version='2.5.4', numpy_version='1.26.4', scipy_version='1.12.0')
            and design.get('candidate_numpy_version') == '2.4.2',
            'registered full-scale runtime stacks changed')
    if candidate_backend == 'cuda':
        arithmetic = design.get('candidate_arithmetic_path', {})
        code = design.get('candidate_code', {})
        # This declaration binds the already frozen NumPy criteria. The launcher
        # separately verifies the inherited file; this comparator does not read it.
        require(design.get('source_executions_preexist') is True
                and design.get('native_reference_preexists') is True
                and design.get('inherited_tolerance_design') == (
                    'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/'
                    'native_full_scale/comparison_design.json')
                and design.get('inherited_tolerance_design_sha256') == (
                    '028b5e64c15ea201603be51caa3390c708efd94bdbab35baf6f8ae211e157267')
                and isinstance(code, dict) and set(code) == {'drive_sha256','executor_sha256','runner_sha256'}
                and all(isinstance(value, str) and len(value) == 64
                        and set(value) <= set('0123456789abcdef') for value in code.values())
                and isinstance(arithmetic, dict)
                and all(arithmetic.get(key) == value for key,value in dict(
                    backend='cuda', numpy_version='2.4.2', state_dtype='float64',
                    torch_version='2.9.0+cu128', cuda_version='12.8', cuda_device='NVIDIA L40S',
                    operation_execution='eager operations; no compiled/fused replacement').items())
                and 'cuda_process_peak_allocated_bytes' not in arithmetic
                and isinstance(design.get('candidate_source_jobs'), list)
                and len(design['candidate_source_jobs']) == 9
                and all(type(job) is int and job > 0 for job in design['candidate_source_jobs'])
                and len(set(design['candidate_source_jobs'])) == 9,
                'registered CUDA arithmetic, code, source jobs or inherited tolerance declaration changed')
    require(factorization_proof is not None, 'full-scale comparison requires archived factorization proof')
    proof_root = Path(factorization_proof)
    proof_path, report_path, prior_design_path, helper_path = (
        proof_root/name for name in ('source_comparison.json', 'factorized/report.json',
                                    'design.json', 'Libs/population_dynamics/pfc_factorized_source.py'))
    require(sha256_file(proof_path) == design.get('reference_factorization_parity_sha256'),
            'registered factorization parity proof changed')
    proof, prior, prior_design = (json.loads(path.read_text()) for path in
                                 (proof_path, report_path, prior_design_path))
    helper_sha = sha256_file(helper_path)
    require(proof.get('status') == 'PASS_SOURCE_UNIFORM_PROJECTION_EQUIVALENCE'
            and proof.get('candidate_report_sha256') == sha256_file(report_path)
            and proof.get('design_sha256') == sha256_file(prior_design_path)
            and helper_sha == design.get('reference_execution_implementation_sha256')
            and proof.get('execution_implementation_sha256') == helper_sha
            and prior.get('execution_implementation_sha256') == helper_sha
            and prior.get('status') == 'NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS'
            and prior.get('synapse_execution') == 'factorized_uniform_projections',
            'factorization helper, prior report or parity proof binding mismatch')
    for record in (proof, prior):
        require(record.get('source_revision') == source.SOURCE_REVISION
                and record.get('source_entry_sha256') == source.SOURCE_ENTRY_SHA256
                and record.get('full_scale_qualified') is False
                and record.get('scientific_fidelity_qualified') is False,
                'factorization proof source identity or scope mismatch')
    prior_config = dict(prior['resolved_config'])
    del prior_config['synapse_execution']
    registered = source.config_contract(prior_design, prior_config, prior['resolved_config'])
    require(len(proof.get('cases', [])) == len(prior.get('cases', [])) == len(registered)
            and {row['case_id'] for row in proof['cases']} == set(registered)
            and {row['case_id'] for row in prior['cases']} == set(registered)
            and all(row.get('spike_arrays_exact') is True
                    and type(row.get('max_voltage_mV_difference')) in (int,float)
                    and 0 <= row['max_voltage_mV_difference'] <= 1e-10 for row in proof['cases'])
            and all(prior.get(key) == design['reference_runtime'][key]
                    for key in ('brian2_version','numpy_version')),
            'complete reference-size factorization parity required')
    return dict(full_scale_design=True,
        reference_synapse_execution=expected['reference_synapse_execution'],
        reference_execution_implementation_sha256=helper_sha,
        reference_clock_correspondence=reference_clock,
        native_config=_native_case_projection(source,config,expected_ids) if reference_clock else config,
        factorization_proof_binding=dict(path=str(proof_root.resolve()),
            comparison_sha256=sha256_file(proof_path), report_sha256=sha256_file(report_path),
            design_sha256=sha256_file(prior_design_path), helper_sha256=helper_sha,
            interpretation='prior 2560-unit uniform-projection parity; does not establish 100k equality'))


def _compare_streamed(candidate, references, design_path, output, config_path, source_runs,
                      factorization_proof=None, *, candidate_kind='streamed_numpy'):
    # This import is intentionally delayed: scale analysis imports this module
    # for the historical Brian2 comparison, which keeps its original dependencies.
    from . import pfc_scale_analysis as source

    require = source.require
    require(config_path is not None and source_runs is not None,
            'streamed comparison requires canonical config and explicit local source archives')
    design = json.loads(design_path.read_text())
    config, registered, seeds, dimensions = source._source_registration(config_path)
    config_sha = sha256_file(config_path)
    candidate_backend = candidate_kind.removeprefix('streamed_')
    contract = _streamed_design_contract(source, design, config, config_sha, factorization_proof,
                                         candidate_backend=candidate_backend)
    canonical_registered = registered
    reference_clock = contract.get('reference_clock_correspondence',False)
    case_ids = design['case_ids'] if reference_clock else [row['case_id'] for row in config['cases']]
    registered = {case_id:canonical_registered[case_id] for case_id in case_ids}
    case_configs = {row['case_id']: row for row in config['cases'] if row['case_id'] in registered}
    if reference_clock:
        native_path = design_path.parent/design['native_config_file']
        require(sha256_file(native_path) == design['native_config_sha256']
                and yaml.safe_load(native_path.read_text()) == contract['native_config'],
                'reference-clock native configuration projection or hash changed')
        seeds = design['source_rng_blocks']
    require(len(source_runs) == len(registered),
            'complete preregistered streamed archive coverage required' if reference_clock
            else 'complete nine-case streamed archive coverage required')

    batch_path, batch_index = None, {}
    if contract['full_scale_design']:
        require(candidate is None, 'full-scale design uses explicit source archives; a batch argument is ambiguous')
    else:
        require(candidate is not None, 'reference-size streamed comparison requires its complete batch')
        batch_path = candidate/'delivery.json'
        batch = json.loads(batch_path.read_text())
        require(batch.get('status') == 'COMPLETE' and batch.get('backend') == 'numpy'
                and batch.get('exit_code') == 0 and batch.get('srun_exit_code') == 0
                and batch.get('scientific_fidelity_qualified') is False and batch.get('production_episodes') == 0
                and batch.get('registered_case_ids') == case_ids
                and len(batch.get('case_ids', [])) == len(registered)
                and set(batch['case_ids']) == set(registered),
                'complete registered streamed batch required')
        batch_cases = batch.get('cases', [])
        require(len(batch_cases) == len(registered)
                and {row.get('case_id') for row in batch_cases} == set(registered),
                'streamed batch case coverage differs from design')
        batch_index = {row['case_id']: row for row in batch_cases}

    index, reference_bindings, native_runtime = {}, [], None
    for root in references:
        report_path = root/'report.json'
        report = json.loads(report_path.read_text())
        require(report.get('status') == 'NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS'
                and report.get('source_revision') == source.SOURCE_REVISION
                and report.get('source_entry_sha256') == source.SOURCE_ENTRY_SHA256
                and report.get('execution_implementation_sha256') == contract['reference_execution_implementation_sha256']
                and report.get('synapse_execution') == contract['reference_synapse_execution']
                and report.get('full_scale_qualified') is False
                and report.get('scientific_fidelity_qualified') is False,
                'complete native reference source identity required')
        runtime = {key: report.get(key) for key in ('brian2_version', 'numpy_version')}
        archive_binding = {}
        if contract['full_scale_design']:
            require(report.get('resolved_config') == contract['native_config'], 'full-scale reference config differs from design')
            archive = root.parent
            delivery_path = archive/'delivery.json'
            manifest = source._source_manifest(json.loads(delivery_path.read_text()))
            bound_report = source._source_bound_file(archive, manifest, root.name+'/report.json')
            helper_path = source._source_bound_file(
                archive, manifest, 'Libs/population_dynamics/pfc_factorized_source.py')
            execution_path = source._source_bound_file(archive, manifest, 'execution.json')
            execution = json.loads(execution_path.read_text())
            require(bound_report.resolve() == report_path.resolve()
                    and sha256_file(helper_path) == contract['reference_execution_implementation_sha256']
                    and execution.get('status') == ('NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE' if reference_clock
                                                    else 'NATIVE_FULL_SCALE_EXECUTION_COMPLETE')
                    and execution.get('exit_code') == 0 and execution.get('completed_cases') == design['cases']
                    and execution.get('native_report_sha256') == sha256_file(report_path),
                    'full-scale reference helper or execution receipt binding mismatch')
            if reference_clock:
                native_file = source._source_bound_file(archive,manifest,'native_config.yaml')
                require(sha256_file(native_file) == design['native_config_sha256']
                        and yaml.safe_load(native_file.read_text()) == contract['native_config']
                        and execution.get('design_sha256') == sha256_file(design_path)
                        and execution.get('native_config_sha256') == design['native_config_sha256']
                        and execution.get('source_config_sha256') == config_sha,
                        'reference-clock native execution preregistration or configuration binding changed')
            versions = execution.get('runtime_versions', {})
            require(all(versions.get(key.removesuffix('_version')) == value
                        for key,value in design['reference_runtime'].items())
                    and all(report.get(key) == design['reference_runtime'][key]
                            for key in ('brian2_version','numpy_version'))
                    and isinstance(versions.get('python'), str) and bool(versions['python']),
                    'full-scale reference runtime receipt differs from registered stack')
            # SciPy and Python are observed in the archived runtime receipt,
            # not added to or inferred from the immutable native report.
            runtime.update(scipy_version=versions['scipy'], python_version=versions['python'])
            archive_binding = dict(delivery_sha256=sha256_file(delivery_path),
                execution_receipt_sha256=sha256_file(execution_path),
                helper_sha256=sha256_file(helper_path),
                runtime_version_sources=dict(brian2='native/report.json and execution.json',
                    numpy='native/report.json and execution.json', scipy='execution.json', python='execution.json'))
        else:
            source.config_contract(design, report['resolved_config'], config)
        require(all(isinstance(value, str) and value for value in runtime.values())
                and (native_runtime is None or runtime == native_runtime),
                'native reference runtime identity missing or inconsistent')
        native_runtime = runtime
        for case in report['cases']:
            case_id = case['case_id']
            require(case_id in registered and case_id not in index,
                    'duplicate or unregistered native reference case')
            require(all(case.get(key) == value for key, value in case_configs[case_id].items())
                    and case.get('core_units') == dimensions['coupled_core_units'],
                    'native reference seed, control or unit identity changed')
            index[case_id] = root, case
        reference_bindings.append(dict(path=str(root.resolve()), report_sha256=sha256_file(report_path),
                                       **archive_binding))
    require(set(index) == set(registered),
            'complete preregistered native reference coverage required' if reference_clock
            else 'complete nine-case native reference coverage required')

    # Admit the entire denominator before opening any scientific arrays. Staging
    # locations are explicit inputs; immutable batch origin paths are never read.
    headers, source_bindings = {}, []
    for local in source_runs:
        status, header = source._source_header(local, config, canonical_registered, config_sha)
        require(header is not None and status['status'] == 'COMPLETE',
                'complete archived streamed case required')
        case_id = header['report']['case_id']
        require(case_id not in headers, 'duplicate streamed source case')
        require(case_id in registered, 'unregistered streamed source case')
        if reference_clock:
            expected_source = next(row for row in design['preexisting_source_bindings'] if row['case_id'] == case_id)
            require(all(header['binding'][key] == expected_source[key]
                        for key in ('report_sha256','delivery_sha256')),
                    'preregistered existing source archive changed')
            require(header['report'].get('backend') == 'numpy'
                    and header['report'].get('driver',{}).get('numpy_version') == design['candidate_numpy_version'],
                    'reference-clock candidate runtime changed')
            source._source_execution_metadata(header,config,canonical_registered,dimensions,config_sha)
        if contract['full_scale_design']:
            # The frozen design supplies the denominator for existing single-case
            # and multi-case archives. Their established admission does not require
            # later launcher fields such as rank or a batch execution receipt.
            require(header['report'].get('backend') == candidate_backend,
                    'full-scale source backend differs from explicit candidate backend')
            delivery = json.loads((header['archive']/'delivery.json').read_text())
            if 'job_id' in delivery:
                header['binding']['job_id'] = delivery['job_id']
            if candidate_backend == 'cuda':
                report = header['report']
                arithmetic = report.get('simulator_cost', {}).get('arithmetic_path', {})
                stable = {key:value for key,value in arithmetic.items()
                          if key != 'cuda_process_peak_allocated_bytes'}
                peak = arithmetic.get('cuda_process_peak_allocated_bytes')
                require(report.get('code') == design['candidate_code'],
                        'CUDA source producer code differs from registered comparison')
                require(stable == design['candidate_arithmetic_path']
                        and report.get('driver', {}).get('numpy_version') == design['candidate_numpy_version']
                        and type(peak) is int and peak >= 0,
                        'CUDA source arithmetic/runtime identity or peak measurement differs from registration')
                require(str(delivery.get('job_id')) == str(design['candidate_source_jobs'][case_ids.index(case_id)]),
                        'CUDA source job differs from registered case archive')

        else:
            row = batch_index[case_id]
            delivery = json.loads((header['archive']/'delivery.json').read_text())
            qualification_sha = header['report']['backend_qualification']['receipt_sha256']
            require(row.get('status') == 'PFC_SOURCE_EXECUTION_COMPLETE'
                    and row.get('source_execution_complete') is True and row.get('backend') == 'numpy'
                    and header['report'].get('backend') == 'numpy'
                    and row.get('exit_code') == delivery.get('exit_code') == 0
                    and row.get('job_id') == delivery.get('job_id') == batch.get('job_id')
                    and type(row.get('rank')) is int and row['rank'] == delivery.get('rank')
                    and row.get('config_sha256') == config_sha
                    and row.get('qualification_sha256') == qualification_sha
                    and row.get('scientific_fidelity_qualified') is False and row.get('production_episodes') == 0
                    and isinstance(row.get('archive'), str) and Path(row['archive']).is_absolute(),
                    'streamed batch and local case archive binding mismatch')
            execution_path = source._source_bound_file(header['archive'], header['manifest'], 'execution.json')
            execution = json.loads(execution_path.read_text())
            execution_keys = ('case_id', 'job_id', 'rank', 'status', 'backend', 'exit_code',
                              'source_execution_complete', 'config_sha256', 'qualification_sha256',
                              'scientific_fidelity_qualified', 'production_episodes', 'python_version')
            require(all(execution.get(key) == row.get(key) for key in execution_keys),
                    'streamed batch differs from hash-bound case execution receipt')
            header['binding'].update(original_archive=row['archive'], job_id=row['job_id'], rank=row['rank'],
                                     execution_receipt_sha256=sha256_file(execution_path))
        headers[case_id] = header
    require(set(headers) == set(registered), 'complete unique streamed source coverage required')

    rows, common, stream_runtime = [], None, None
    for case_id in case_ids:
        header = headers[case_id]
        raw, binding, _ = source._source_values(header, config, canonical_registered, dimensions, config_sha)
        require(common is None or common == binding, 'streamed cases use different execution stacks')
        common = binding
        report = header['report']
        arithmetic = report.get('simulator_cost', {}).get('arithmetic_path', {})
        runtime = dict(backend=report['backend'], numpy_version=report['driver']['numpy_version'],
                       arithmetic_path=arithmetic, python_version=batch_index.get(case_id, {}).get('python_version'))
        if candidate_backend == 'cuda':
            # Peak allocation is a case cost measurement, not library or arithmetic
            # identity. Keep it below in every case instead of comparing it across cases.
            runtime['arithmetic_path'] = {key:value for key,value in arithmetic.items()
                                         if key != 'cuda_process_peak_allocated_bytes'}
            runtime['scipy_version'] = None
            runtime.update({key:arithmetic[key] for key in ('torch_version','cuda_version','cuda_device')})
        require(arithmetic.get('backend') == candidate_backend
                and arithmetic.get('numpy_version') == runtime['numpy_version']
                and arithmetic.get('state_dtype') == 'float64'
                and (stream_runtime is None or runtime == stream_runtime),
                'streamed simulator arithmetic/runtime identity mismatch')
        if contract['full_scale_design']:
            require(runtime['numpy_version'] == design['candidate_numpy_version'],
                    'full-scale candidate NumPy runtime differs from registered stack')
        stream_runtime = runtime
        root, native_case = index[case_id]
        if reference_clock:
            other = source.raw_values(root,native_case,unit_count=dimensions['coupled_core_units'],
                                      dt_ms=design['dt_ms'],duration_ticks=dimensions['duration_ticks'])
        else:
            other = source.raw_values(root,native_case,unit_count=dimensions['coupled_core_units'])
        arrays = _array_diagnostics(raw, other)
        failures = [row['array'] for row in arrays if not row['passed']]
        voltage = next(row for row in arrays if row['array'] == 'terminal_voltage_mV')
        rows.append(dict(case_id=case_id, **{k:v for k,v in case_configs[case_id].items() if k != 'case_id'},
            status='FAILED_NUMERICAL_EQUIVALENCE' if failures else 'PASS', failed_arrays=failures,
            spike_arrays_exact=all(row['passed'] for row in arrays if row['array'] != 'terminal_voltage_mV'),
            max_voltage_mV_difference=voltage['max_voltage_mV_difference'],
            candidate_events=report['neural_event_count'], native_events=native_case['native_events'],
            values=sum(row['candidate_values'] for row in arrays),
            reference_values=sum(row['reference_values'] for row in arrays),
            values_compared=sum(row['values_compared'] for row in arrays), arrays=arrays,
            candidate_seconds=report['costs']['total_execute_wall_seconds'],
            native_seconds=native_case['elapsed_seconds']))
        if candidate_backend == 'cuda':
            rows[-1]['cuda_process_peak_allocated_bytes'] = arithmetic['cuda_process_peak_allocated_bytes']
        header['binding'].update(raw_sha256=report['outputs']['raw.npz']['sha256'],
                                 native_raw_sha256=native_case['raw_sha256'])
        source_bindings.append(header['binding'])

    failures = [row['case_id'] for row in rows if row['status'] != 'PASS']
    voltage_deltas = [row['max_voltage_mV_difference'] for row in rows]
    result = dict(status='FAILED_STREAMED_SOURCE_EQUIVALENCE' if failures else 'PASS_STREAMED_SOURCE_EQUIVALENCE',
        candidate_kind=candidate_kind, cases=rows, registered_cases=len(registered), compared_cases=len(rows),
        passed_cases=len(rows)-len(failures), failed_case_ids=failures,
        array_values=sum(row['values'] for row in rows),
        reference_array_values=sum(row['reference_values'] for row in rows),
        array_values_compared=sum(row['values_compared'] for row in rows),
        max_voltage_mV_difference=max(voltage_deltas) if all(v is not None for v in voltage_deltas) else None,
        reference_bindings=reference_bindings, source_bindings=source_bindings,
        config_sha256=config_sha,
        design_sha256=sha256_file(design_path), comparison_implementation_sha256=sha256_file(Path(__file__)),
        source_admission_implementation_sha256=sha256_file(Path(source.__file__)),
        source_revision=source.SOURCE_REVISION, source_entry_sha256=source.SOURCE_ENTRY_SHA256,
        execution_implementation_sha256=common['code']['executor_sha256'],
        common_execution_binding=common, source_rng_seeds=seeds,
        tolerances=dict(relative=0., terminal_voltage_mV_absolute=1e-10, other_arrays='exact'),
        runtime_stacks=dict(native=dict(backend=contract['reference_synapse_execution'], **native_runtime),
            candidate=stream_runtime,
            numpy_versions_equal=native_runtime['numpy_version'] == stream_runtime['numpy_version'],
            interpretation='Comparison of the combined execution and numerical-library stacks; failure does not '
                           'isolate backend implementation from numerical-library versions.'),
        full_scale_qualified=False, scientific_fidelity_qualified=False, working_memory_qualified=False,
        fidelity_at_requested_scale_qualified=False, learned_simulator=False, production_episodes=0,
        cost_interpretation='separate executions; reported execute times exclude staging and scheduler costs; '
                            'not a matched scientific speedup')
    if batch_path is not None:
        result['candidate_batch_delivery_sha256'] = sha256_file(batch_path)
    if contract['full_scale_design']:
        result.update(comparison_scope=design['scope'],
                      reference_factorization_proof=contract['factorization_proof_binding'],
                      registered_runtime_stacks=dict(reference=design['reference_runtime'],
                          candidate=dict(backend=design['candidate_backend'],
                                         numpy_version=design['candidate_numpy_version'])))
        if candidate_backend == 'cuda':
            result['registered_runtime_stacks']['candidate']['arithmetic_path'] = design['candidate_arithmetic_path']
            result['registered_candidate_code'] = design['candidate_code']
            result['inherited_tolerance_design_sha256'] = design['inherited_tolerance_design_sha256']
    if reference_clock:
        result.update(status='FAILED_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE' if failures
                             else 'PASS_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE',
            canonical_registered_cases=len(canonical_registered),
            canonical_case_ids=[case['case_id'] for case in config['cases']],
            uncompared_canonical_case_ids=[case['case_id'] for case in config['cases'] if case['case_id'] not in registered],
            native_config_sha256=design['native_config_sha256'],native_configuration_projection='cases_only')
    freeze_json(output, result)
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == 'prepare-reference':
        parser = argparse.ArgumentParser(description='Preregister full-scale reference-clock correspondence')
        for name in ('config','factorization-proof','output'):
            parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--source-run',type=Path,action='append',required=True)
        parser.add_argument('--case-id',action='append',required=True)
        args = parser.parse_args(arguments[1:])
        report = prepare_reference(args.config,args.source_run,args.factorization_proof,args.output,args.case_id)
        print(json.dumps(dict(status=report['status'],scope=report['scope'],cases=report['cases'],
                              native_config=str(args.output/'native_config.yaml'),design=str(args.output/'design.json'))))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path,
                        help='Brian2 candidate or reference-size streamed batch; omit for the full-scale design')
    parser.add_argument('--reference', type=Path, action='append', required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-kind', choices=('brian2', 'streamed_numpy', 'streamed_cuda'), default='brian2')
    parser.add_argument('--config', type=Path, help='Canonical streamed source config, with original bytes')
    parser.add_argument('--source-run', type=Path, action='append',
                        help='Local archived case root or source_run directory; repeat for every streamed case')
    parser.add_argument('--factorization-proof', type=Path,
                        help='Archived reference_size proof root; required only by the full-scale design')
    args = parser.parse_args(arguments)
    report = compare(args.candidate, args.reference, args.design, args.output,
                     candidate_kind=args.candidate_kind, config_path=args.config,
                     source_runs=args.source_run, factorization_proof=args.factorization_proof)
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ('cases', 'reference_bindings', 'source_bindings')}))
    return int(report['status'] in ('FAILED_STREAMED_SOURCE_EQUIVALENCE','FAILED_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE'))


if __name__ == '__main__':
    raise SystemExit(main())
