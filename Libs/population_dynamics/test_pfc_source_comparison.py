"""Streamed comparison tests; synthetic arrays are not scientific evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from . import pfc_source_comparison as comparison
from . import pfc_scale_analysis as admission
from . import test_pfc_source_scale_analysis as source_fixture
from .provenance import sha256_file


# The archive fixture's nonexecutable legacy hash allowance is process-local.
# Carry it into subprocess tests without changing the production whitelist.
_FIXTURE_CLI_BOOTSTRAP = (
    "import hashlib,runpy\n"
    "from population_dynamics import pfc_scale_analysis as admission\n"
    "fixture=hashlib.sha256(b'# Engineering fixture, not executable source: pfc_deterministic_replay.py\\n').hexdigest()\n"
    "admission._LEGACY_DEFAULT_KERNEL_EXECUTORS=admission._LEGACY_DEFAULT_KERNEL_EXECUTORS | {fixture}\n"
    "runpy.run_module('population_dynamics.pfc_source_comparison',run_name='__main__')\n"
)


class StreamedSourceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = source_fixture.SourceReadoutTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config_path = self.fixture.config_path
        canonical = Path(__file__).parents[1]/'configs/neural_population_pfc_reference_size_factorized.yaml'
        self.config_path.write_bytes(canonical.read_bytes())
        self.config = self.fixture.config = yaml.safe_load(self.config_path.read_text())
        _, self.fixture.registered, self.fixture.seeds, self.fixture.dimensions = admission._source_registration(
            self.config_path)
        self.native = self.root/'native'
        self.native.mkdir()
        self.batch = self.root/'batch'
        self.batch.mkdir()
        self.sources, native_cases, batch_cases = [], [], []
        for rank, case in enumerate(self.config['cases']):
            source = self.fixture.archive(case['case_id'])
            self.expand_source(source)
            self.sources.append(source)
            report = admission.read(source/'report.json')
            execution = dict(case_id=case['case_id'], job_id='engineering-fixture', rank=rank,
                status='PFC_SOURCE_EXECUTION_COMPLETE', backend='numpy', exit_code=0,
                source_execution_complete=True, config_sha256=sha256_file(self.config_path),
                qualification_sha256=report['backend_qualification']['receipt_sha256'],
                scientific_fidelity_qualified=False, production_episodes=0,
                python_version='engineering-fixture')
            self.write(source.parent/'execution.json', execution)
            delivery = admission.read(source.parent/'delivery.json')
            delivery.update(job_id=execution['job_id'], rank=rank, exit_code=0)
            delivery['files'].append(self.fixture.file_row(source.parent, 'execution.json'))
            self.write(source.parent/'delivery.json', delivery)
            # These origins intentionally do not exist: local copies must suffice.
            batch_cases.append(dict(execution, archive='/unavailable/original/'+case['case_id']))
            raw = self.native/(case['case_id']+'_native.npz')
            shutil.copyfile(source/'raw.npz', raw)
            native_cases.append(dict(case, core_units=2560, native_events=report['neural_event_count'],
                                     raw_file=raw.name, raw_sha256=sha256_file(raw), elapsed_seconds=2.))
        native_config = dict(self.config)
        del native_config['synapse_execution']
        self.write(self.native/'report.json', dict(status='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',
            source_revision=admission.SOURCE_REVISION, source_entry_sha256=admission.SOURCE_ENTRY_SHA256,
            execution_implementation_sha256=admission.SOURCE_ENTRY_SHA256, synapse_execution='native_source',
            full_scale_qualified=False, scientific_fidelity_qualified=False, resolved_config=native_config,
            cases=native_cases, brian2_version='2.5.4', numpy_version='1.26.4'))
        case_ids = [case['case_id'] for case in self.config['cases']]
        self.write(self.batch/'delivery.json', dict(status='COMPLETE', backend='numpy',
            job_id='engineering-fixture', exit_code=0, srun_exit_code=0,
            scientific_fidelity_qualified=False, production_episodes=0,
            case_ids=case_ids, registered_case_ids=case_ids, cases=batch_cases))
        self.design = self.root/'design.json'
        self.write(self.design, dict(status='REGISTERED_BEFORE_SOURCE_EXECUTION',
            scope='bounded_source_scaling_diagnostic', source_rng_blocks=[93001,93002,93003],
            training_seeds=[41,42,43], cases=9, duration_ms=5000., units=2560, phases_ms=admission.PHASES,
            relative_tolerance=0., terminal_voltage_mV_max_absolute_error=1e-10,
            required_event_and_integer_arrays='exact', production_episodes=0,
            full_scale_qualified=False, scientific_fidelity_qualified=False))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')

    @staticmethod
    def arrays(path):
        with np.load(path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}

    def expand_source(self, source):
        """Reuse the established archive fixture at the required reference unit count."""
        n_e, n_i = self.config['excitatory_neurons'], self.config['inhibitory_neurons']
        total = n_e+n_i
        raw = self.arrays(source/'raw.npz')
        raw['native_spike_counts'] = np.bincount(raw['spike_unit_indices'], minlength=total)
        old = self.arrays(source/'terminal_state.npz')
        terminal = {}
        for pop, size, offset in (('E',n_e,0), ('I',n_i,n_e)):
            for name, value in old.items():
                if name.startswith(pop+'__'):
                    terminal[name] = np.zeros(size, dtype=value.dtype)
            terminal[pop+'__v'][:] = -.06
            terminal[pop+'__not_refractory'][:] = True
            terminal[pop+'__source_indices'] = np.arange(size, dtype=np.int64)
            terminal[pop+'__archive_indices'] = np.arange(size, dtype=np.int64)+offset
            terminal[pop+'__spike_counts'] = raw['native_spike_counts'][offset:offset+size]
        raw['terminal_voltage_mV'] = np.r_[terminal['E__v'],terminal['I__v']]/.001
        np.savez(source/'raw.npz', **raw)
        np.savez(source/'terminal_state.npz', **terminal)
        np.savez(source/'units.npz', unit_ids=np.arange(total, dtype=np.int64),
            source_unit_indices=np.r_[np.arange(n_e,dtype=np.int64),np.arange(n_i,dtype=np.int64)],
            population_index=np.r_[np.zeros(n_e,dtype=np.uint8),np.ones(n_i,dtype=np.uint8)],
            excitatory_preference_unit_ids=np.arange(n_e,dtype=np.int64),
            excitatory_preferred_angle_deg=360.*np.arange(n_e,dtype=np.float64)/n_e)
        seed = admission.read(source/'report.json')['seeds']['source_rng_seed']
        np.save(source/'initial_voltage.npy', np.full(total,-60.+seed/1e6,dtype=np.float64))
        metadata = admission.read(source/'unit_metadata.json')
        metadata['unit_count'] = total
        metadata['coupling']['normalization'] = self.config['weight_scaling_factor']
        self.write(source/'unit_metadata.json', metadata)
        report = admission.read(source/'report.json')
        report['driver'].update(n_e=n_e, n_i=n_i)
        report['inputs_used']['E']['values'] = n_e*100000
        report['inputs_used']['I']['values'] = n_i*100000
        report['costs'] = dict(total_execute_wall_seconds=1.)
        report['simulator_cost'] = dict(arithmetic_path=dict(
            backend='numpy', numpy_version=report['driver']['numpy_version'], state_dtype='float64'))
        self.write(source/'report.json', report)
        self.fixture.rebind(source)

    def run_comparison(self, name='comparison.json', sources=None):
        return comparison.compare(self.batch, [self.native], self.design, self.root/name,
            candidate_kind='streamed_numpy', config_path=self.config_path,
            source_runs=self.sources if sources is None else sources)

    def test_complete_relocated_batch_keeps_actual_runtime_stacks_and_receipts(self):
        paths = [self.batch/'delivery.json', *[source.parent/'delivery.json' for source in self.sources],
                 *[source/'report.json' for source in self.sources]]
        before = [sha256_file(path) for path in paths]
        result = self.run_comparison(sources=list(reversed(self.sources)))
        self.assertEqual(result['status'], 'PASS_STREAMED_SOURCE_EQUIVALENCE')
        self.assertEqual((result['registered_cases'], result['compared_cases'], result['passed_cases']), (9,9,9))
        self.assertEqual(len(result['source_bindings']), 9)
        self.assertEqual(result['source_rng_seeds'], [93001,93002,93003])
        stacks = result['runtime_stacks']
        self.assertEqual(stacks['native']['numpy_version'], '1.26.4')
        self.assertEqual(stacks['candidate']['numpy_version'], 'engineering_fixture')
        self.assertNotIn('brian2_version', stacks['candidate'])
        self.assertFalse(stacks['numpy_versions_equal'])
        self.assertIn('does not isolate', stacks['interpretation'])
        self.assertFalse(result['full_scale_qualified'])
        self.assertFalse(result['scientific_fidelity_qualified'])
        self.assertEqual(result['production_episodes'], 0)
        self.assertEqual(before, [sha256_file(path) for path in paths])
        self.assertTrue(all(row['original_archive'].startswith('/unavailable/original/')
                            for row in result['source_bindings']))

    def test_default_brian2_behavior_retains_equal_version_requirement(self):
        candidate = self.root/'brian2'
        shutil.copytree(self.native, candidate)
        result = comparison.compare(candidate, [self.native], self.design, self.root/'brian2_pass.json')
        self.assertEqual(result['status'], 'PASS_SOURCE_UNIFORM_PROJECTION_EQUIVALENCE')
        report = admission.read(candidate/'report.json')
        report['numpy_version'] = 'different'
        self.write(candidate/'report.json', report)
        with self.assertRaisesRegex(ValueError, 'PFC_REFERENCE_SOURCE_MISMATCH:numpy_version'):
            comparison.compare(candidate, [self.native], self.design, self.root/'brian2_rejected.json')

    def test_cli_preserves_first_and_last_case_failures_and_exits_nonzero(self):
        report = admission.read(self.native/'report.json')
        for number in (0,8):
            case = report['cases'][number]
            path = self.native/case['raw_file']
            raw = self.arrays(path)
            if number == 0:
                raw['terminal_voltage_mV'][0] += 1e-7
            else:
                raw['spike_ticks'] = np.r_[raw['spike_ticks'], np.int64(90000)]
                raw['spike_unit_indices'] = np.r_[raw['spike_unit_indices'], np.int64(13)]
                raw['spike_seconds'] = raw['spike_ticks']*5e-5
                raw['native_spike_counts'][13] += 1
                case['native_events'] += 1
            np.savez(path, **raw)
            case['raw_sha256'] = sha256_file(path)
        self.write(self.native/'report.json', report)
        output = self.root/'cli_failure.json'
        args = [sys.executable,'-B','-c',_FIXTURE_CLI_BOOTSTRAP,
                '--candidate-kind', 'streamed_numpy', '--candidate', str(self.batch),
                '--config', str(self.config_path), '--reference', str(self.native),
                '--design', str(self.design), '--output', str(output)]
        for source in self.sources:
            args.extend(['--source-run',str(source)])
        completed = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, completed.stdout+completed.stderr)
        result = admission.read(output)
        self.assertEqual(result['status'], 'FAILED_STREAMED_SOURCE_EQUIVALENCE')
        self.assertEqual(result['compared_cases'], 9)
        self.assertEqual(result['passed_cases'], 7)
        self.assertEqual(result['failed_case_ids'], [report['cases'][i]['case_id'] for i in (0,8)])
        self.assertEqual(result['cases'][0]['failed_arrays'], ['terminal_voltage_mV'])
        self.assertIn('spike_ticks', result['cases'][-1]['failed_arrays'])
        self.assertTrue(all(len(row['arrays']) == 5 for row in result['cases']))

    def test_incomplete_duplicate_and_wrong_batch_coverage_reject_before_arrays(self):
        for name, sources in [('missing',self.sources[:-1]),
                              ('duplicate',[self.sources[0], *self.sources[:-1]])]:
            with self.subTest(name=name), patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
                with self.assertRaises(ValueError):
                    self.run_comparison(name+'.json', sources)
        batch = admission.read(self.batch/'delivery.json')
        batch['cases'].pop()
        self.write(self.batch/'delivery.json', batch)
        with patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
            with self.assertRaisesRegex(ValueError, 'batch case coverage'):
                self.run_comparison()
        self.assertFalse((self.root/'comparison.json').exists())

    def test_fixed_tolerances_cannot_be_relaxed(self):
        original = admission.read(self.design)
        for key,value in [('relative_tolerance',1e-10), ('terminal_voltage_mV_max_absolute_error',1e-9)]:
            self.write(self.design, dict(original, **{key:value}))
            with patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
                with self.assertRaisesRegex(ValueError, 'registered reference-size'):
                    self.run_comparison()

    def test_batch_and_hash_bound_execution_receipt_must_agree(self):
        batch = admission.read(self.batch/'delivery.json')
        batch['cases'][0]['python_version'] = 'unbound replacement'
        self.write(self.batch/'delivery.json', batch)
        with patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
            with self.assertRaisesRegex(ValueError, 'hash-bound case execution'):
                self.run_comparison()

    def test_corrupt_raw_payload_fails_closed(self):
        with (self.sources[0]/'raw.npz').open('ab') as stream:
            stream.write(b'corrupted')
        with patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
            with self.assertRaisesRegex(ValueError, 'size or hash mismatch'):
                self.run_comparison()
        self.assertFalse((self.root/'comparison.json').exists())

    def test_rebound_source_seed_and_control_changes_fail_before_arrays(self):
        path = self.sources[0]/'report.json'
        original = admission.read(path)
        for key,value in [('source_rng_seed',93002), ('stimulus_strength_nA',.5)]:
            report = copy.deepcopy(original)
            report['selected_case'][key] = value
            self.write(path, report)
            self.fixture.rebind(self.sources[0])
            with patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
                with self.assertRaisesRegex(ValueError, 'selected source case identity'):
                    self.run_comparison()

    def test_rebound_unit_identity_and_runtime_disagreement_fail_closed(self):
        source = self.sources[0]
        units = self.arrays(source/'units.npz')
        units['unit_ids'][0] = 1
        np.savez(source/'units.npz', **units)
        self.fixture.rebind(source)
        with self.assertRaisesRegex(ValueError, 'full-unit identity'):
            self.run_comparison('units.json')
        units['unit_ids'][0] = 0
        np.savez(source/'units.npz', **units)
        report = admission.read(source/'report.json')
        report['simulator_cost']['arithmetic_path']['numpy_version'] = 'different'
        self.write(source/'report.json', report)
        self.fixture.rebind(source)
        with self.assertRaisesRegex(ValueError, 'arithmetic/runtime identity'):
            self.run_comparison('runtime.json')

    def test_native_unit_identity_and_source_revision_are_checked(self):
        path = self.native/'report.json'
        original = admission.read(path)
        for kind in ('units','source'):
            report = copy.deepcopy(original)
            if kind == 'units':
                report['cases'][0]['core_units'] = 2561
            else:
                report['source_revision'] = 'different-source'
            self.write(path, report)
            with patch.object(admission.np, 'load', side_effect=AssertionError('arrays opened')):
                with self.assertRaises(ValueError):
                    self.run_comparison(kind+'.json')



class FullScaleComparisonContractTests(unittest.TestCase):
    """Metadata admission and a two-event array fixture, not full-scale scientific evidence."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='pfc_full_contract_')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        libs = Path(__file__).parents[1]
        self.config_path = libs/'configs/neural_population_pfc_100k_source.yaml'
        self.config, _, _, _ = admission._source_registration(self.config_path)
        self.proof = self.root/'proof'
        (self.proof/'factorized').mkdir(parents=True)
        helper_name = 'Libs/population_dynamics/pfc_factorized_source.py'
        self.proof_helper = self.proof/helper_name
        self.proof_helper.parent.mkdir(parents=True)
        self.proof_helper.write_bytes((libs/'population_dynamics/pfc_factorized_source.py').read_bytes())
        helper_sha = sha256_file(self.proof_helper)
        old_config = yaml.safe_load((libs/'configs/neural_population_pfc_reference_size_factorized.yaml').read_text())
        prior_design = dict(status='REGISTERED_BEFORE_SOURCE_EXECUTION', scope='bounded_source_scaling_diagnostic',
            source_rng_blocks=[93001,93002,93003], training_seeds=[41,42,43], cases=9, duration_ms=5000.,
            units=2560, phases_ms=admission.PHASES, relative_tolerance=0.,
            terminal_voltage_mV_max_absolute_error=1e-10, required_event_and_integer_arrays='exact',
            production_episodes=0, full_scale_qualified=False, scientific_fidelity_qualified=False)
        self.write(self.proof/'design.json', prior_design)
        prior = dict(status='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS', resolved_config=old_config,
            source_revision=admission.SOURCE_REVISION, source_entry_sha256=admission.SOURCE_ENTRY_SHA256,
            execution_implementation_sha256=helper_sha, synapse_execution='factorized_uniform_projections',
            brian2_version='2.5.4', numpy_version='1.26.4', full_scale_qualified=False,
            scientific_fidelity_qualified=False, cases=[dict(case, core_units=2560) for case in old_config['cases']])
        self.write(self.proof/'factorized/report.json', prior)
        self.write(self.proof/'source_comparison.json', dict(status='PASS_SOURCE_UNIFORM_PROJECTION_EQUIVALENCE',
            source_revision=admission.SOURCE_REVISION, source_entry_sha256=admission.SOURCE_ENTRY_SHA256,
            execution_implementation_sha256=helper_sha,
            candidate_report_sha256=sha256_file(self.proof/'factorized/report.json'),
            design_sha256=sha256_file(self.proof/'design.json'), full_scale_qualified=False,
            scientific_fidelity_qualified=False, cases=[dict(case, spike_arrays_exact=True,
                max_voltage_mV_difference=0.) for case in old_config['cases']]))
        self.design = dict(status='REGISTERED_BEFORE_SOURCE_EXECUTION',
            scope='native_full_scale_source_equivalence', cases=9, units=100000,
            excitatory_neurons=80000, inhibitory_neurons=20000, weight_scaling_factor=.0256,
            dt_ms=.05, duration_ms=5000., source_rng_blocks=[94001,94002,94003],
            case_ids=[case['case_id'] for case in self.config['cases']],
            source_config_sha256=sha256_file(self.config_path), source_revision=admission.SOURCE_REVISION,
            source_entry_sha256=admission.SOURCE_ENTRY_SHA256,
            reference_synapse_execution='factorized_uniform_projections',
            reference_execution_implementation_sha256=helper_sha,
            reference_factorization_parity_sha256=sha256_file(self.proof/'source_comparison.json'),
            relative_tolerance=0., terminal_voltage_mV_max_absolute_error=1e-10,
            required_event_and_integer_arrays='exact', candidate_backend='numpy', candidate_numpy_version='2.4.2',
            reference_runtime=dict(brian2_version='2.5.4', numpy_version='1.26.4', scipy_version='1.12.0'),
            all_case_numerical_failures_retained=True, full_scale_qualified=False,
            scientific_fidelity_qualified=False, production_episodes=0)
        self.design_path = self.root/'comparison_design.json'
        self.write(self.design_path, self.design)
        self.archive = self.root/'reference_archive'
        self.native = self.archive/'native'
        self.native.mkdir(parents=True)
        self.native_helper = self.archive/helper_name
        self.native_helper.parent.mkdir(parents=True)
        shutil.copyfile(self.proof_helper, self.native_helper)
        report = dict(prior, resolved_config=self.config,
            cases=[dict(case, core_units=100000) for case in self.config['cases']])
        self.write(self.native/'report.json', report)
        self.write(self.archive/'execution.json', dict(status='NATIVE_FULL_SCALE_EXECUTION_COMPLETE',
            exit_code=0, completed_cases=9, native_report_sha256=sha256_file(self.native/'report.json'),
            runtime_versions=dict(python='engineering fixture', brian2='2.5.4', numpy='1.26.4', scipy='1.12.0')))
        self.rebind_reference()
        self.sources = []
        for index, case in enumerate(self.config['cases']):
            root = self.root/'source_archives'/case['case_id']
            source = root/'source_run'
            source.mkdir(parents=True)
            report = dict(status='PFC_SOURCE_EXECUTION_COMPLETE', source_execution_complete=True,
                backend='numpy', case_id=case['case_id'], resolved_config=self.config, selected_case=case,
                config_sha256=sha256_file(self.config_path), scientific_fidelity_qualified=False,
                production_episodes=0)
            self.write(source/'report.json', report)
            delivery = dict(status='ARCHIVED_PFC_SOURCE_EXECUTION', result_status=report['status'],
                source_execution_complete=True, case_id=case['case_id'], scientific_fidelity_qualified=False,
                production_episodes=0, job_id='engineering_single' if index == 0 else 'engineering_packed',
                files=[self.file_row(root, 'source_run/report.json')])
            if index:
                delivery.update(rank=index, exit_code=0)
            # Legacy metadata is deliberately missing rank and execution.json.
            self.write(root/'delivery.json', delivery)
            self.sources.append(source)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')

    @staticmethod
    def file_row(root, name):
        path = root/name
        return dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))

    def rebind_reference(self):
        self.write(self.archive/'delivery.json', dict(files=[self.file_row(self.archive, name) for name in
            ('native/report.json','execution.json','Libs/population_dynamics/pfc_factorized_source.py')]))

    def contract(self, *, design=None, config=None, config_sha=None, proof=True):
        return comparison._streamed_design_contract(admission, self.design if design is None else design,
            self.config if config is None else config,
            sha256_file(self.config_path) if config_sha is None else config_sha,
            self.proof if proof else None)

    def run_metadata(self, *, candidate=None, sources=None, candidate_kind='streamed_numpy'):
        return comparison.compare(candidate, [self.native], self.design_path, self.root/'comparison.json',
            candidate_kind=candidate_kind, config_path=self.config_path,
            source_runs=self.sources if sources is None else sources, factorization_proof=self.proof)

    arrays = staticmethod(StreamedSourceComparisonTests.arrays)

    def prepare_cuda(self, *, complete=False):
        """Adapt the existing source fixture; no CUDA device or simulation is used."""
        if complete:
            self.fixture = source_fixture.SourceReadoutTests()
            self.fixture.setUp()
            self.addCleanup(self.fixture.doCleanups)
            self.fixture.config_path, self.fixture.config = self.config_path, self.config
            _, self.fixture.registered, self.fixture.seeds, self.fixture.dimensions = admission._source_registration(
                self.config_path)
            self.sources = [self.fixture.archive(case['case_id']) for case in self.config['cases']]
            for source in self.sources:
                StreamedSourceComparisonTests.expand_source(self, source)
        arithmetic = dict(backend='cuda', numpy_version='2.4.2', state_dtype='float64',
            torch_version='2.9.0+cu128', cuda_version='12.8', cuda_device='NVIDIA L40S',
            operation_execution='eager operations; no compiled/fused replacement',
            fft='engineering fixture', summed_nmda='engineering fixture')
        code = admission.read(self.sources[0]/'report.json').get('code',
            dict(drive_sha256='a'*64, executor_sha256='b'*64, runner_sha256='c'*64))
        jobs = list(range(700001,700010))
        self.design.update(status='REGISTERED_BEFORE_ARCHIVED_SOURCE_COMPARISON', candidate_backend='cuda',
            candidate_arithmetic_path=arithmetic, candidate_code=code, candidate_source_jobs=jobs,
            source_executions_preexist=True, native_reference_preexists=True,
            inherited_tolerance_design='Output/neural_population_dynamics/multicircuit/'
                'pfc_source_reproduction/native_full_scale/comparison_design.json',
            inherited_tolerance_design_sha256='028b5e64c15ea201603be51caa3390c708efd94bdbab35baf6f8ae211e157267')
        for index, source in enumerate(self.sources):
            report = admission.read(source/'report.json')
            report.update(backend='cuda', code=code)
            report.setdefault('driver', {})['numpy_version'] = '2.4.2'
            report['simulator_cost'] = dict(arithmetic_path=dict(arithmetic,
                cuda_process_peak_allocated_bytes=1024*(index+1)))
            if complete:
                qroot = source.parent/'qualification'
                receipt = admission.read(qroot/'comparison/qualification.json')
                receipt.update(backend='cuda', numpy_version='2.4.2')
                self.write(qroot/'comparison/qualification.json', receipt)
                delivery = admission.read(qroot/'delivery.json')
                delivery['files'] = [self.file_row(qroot, 'comparison/qualification.json')]
                self.write(qroot/'delivery.json', delivery)
                report['backend_qualification'].update(qualification=receipt,
                    receipt_sha256=sha256_file(qroot/'comparison/qualification.json'),
                    delivery_sha256=sha256_file(qroot/'delivery.json'))
                request = admission.read(source/'request.json')
                request.update(backend='cuda', backend_qualification=report['backend_qualification'])
                self.write(source/'request.json', request)
            self.write(source/'report.json', report)
            if complete:
                self.fixture.rebind(source)
            delivery = admission.read(source.parent/'delivery.json')
            delivery.update(job_id=jobs[index])
            for row in delivery['files']:
                if row['path'] == 'source_run/report.json':
                    row.update(self.file_row(source.parent, 'source_run/report.json'))
            self.write(source.parent/'delivery.json', delivery)
        self.write(self.design_path, self.design)
        if complete:
            report = admission.read(self.native/'report.json')
            for case, source in zip(report['cases'], self.sources):
                raw = self.native/(case['case_id']+'_native.npz')
                shutil.copyfile(source/'raw.npz', raw)
                case.update(raw_file=raw.name, raw_sha256=sha256_file(raw), elapsed_seconds=2.,
                            native_events=admission.read(source/'report.json')['neural_event_count'])
            self.write(self.native/'report.json', report)
            self.bind_native_report()

    def bind_native_report(self):
        execution = admission.read(self.archive/'execution.json')
        execution['native_report_sha256'] = sha256_file(self.native/'report.json')
        self.write(self.archive/'execution.json', execution)
        self.rebind_reference()

    def test_cuda_complete_archives_keep_variable_peaks_and_cli_retains_all_numerical_failures(self):
        self.prepare_cuda(complete=True)
        result = self.run_metadata(candidate_kind='streamed_cuda')
        self.assertEqual(result['status'], 'PASS_STREAMED_SOURCE_EQUIVALENCE')
        self.assertEqual(result['candidate_kind'], 'streamed_cuda')
        self.assertEqual((result['registered_cases'],result['compared_cases'],result['passed_cases']), (9,9,9))
        runtime = result['runtime_stacks']['candidate']
        self.assertEqual(runtime['arithmetic_path'], self.design['candidate_arithmetic_path'])
        self.assertEqual(runtime['torch_version'], '2.9.0+cu128')
        self.assertIsNone(runtime['python_version'])
        self.assertIsNone(runtime['scipy_version'])
        self.assertEqual([row['cuda_process_peak_allocated_bytes'] for row in result['cases']],
                         [1024*(index+1) for index in range(9)])
        self.assertEqual(result['registered_candidate_code'], self.design['candidate_code'])
        self.assertFalse(result['scientific_fidelity_qualified'])
        self.assertFalse(result['full_scale_qualified'])
        report = admission.read(self.native/'report.json')
        for index in (0,8):
            case = report['cases'][index]
            path = self.native/case['raw_file']
            raw = self.arrays(path)
            if index == 0:
                raw['terminal_voltage_mV'][0] += 1e-7
            else:
                raw['spike_ticks'] = np.r_[raw['spike_ticks'],np.int64(90000)]
                raw['spike_unit_indices'] = np.r_[raw['spike_unit_indices'],np.int64(99999)]
                raw['spike_seconds'] = raw['spike_ticks']*5e-5
                raw['native_spike_counts'][99999] += 1
                case['native_events'] += 1
            np.savez(path, **raw)
            case['raw_sha256'] = sha256_file(path)
        self.write(self.native/'report.json', report)
        self.bind_native_report()
        output = self.root/'cuda_cli_failure.json'
        args = [sys.executable,'-B','-c',_FIXTURE_CLI_BOOTSTRAP,
            '--candidate-kind','streamed_cuda','--reference',str(self.native),'--design',str(self.design_path),
            '--config',str(self.config_path),'--factorization-proof',str(self.proof),'--output',str(output)]
        for source in self.sources:
            args.extend(['--source-run',str(source)])
        completed = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1, completed.stdout+completed.stderr)
        result = admission.read(output)
        self.assertEqual(result['status'], 'FAILED_STREAMED_SOURCE_EQUIVALENCE')
        self.assertEqual((result['compared_cases'],result['passed_cases']), (9,7))
        self.assertEqual(result['failed_case_ids'], [report['cases'][index]['case_id'] for index in (0,8)])
        self.assertEqual(result['cases'][0]['failed_arrays'], ['terminal_voltage_mV'])
        self.assertIn('spike_ticks', result['cases'][-1]['failed_arrays'])
        self.assertTrue(all(len(row['arrays']) == 5 for row in result['cases']))

    def test_cuda_rejects_backend_version_config_code_and_missing_cases_before_arrays(self):
        self.prepare_cuda()
        source = self.sources[0]
        original = admission.read(source/'report.json')
        changes = [('backend','numpy'), ('torch_version','wrong-version'),
                   ('config_sha256','wrong-config'), ('code','wrong-code')]
        for name,value in changes:
            report = copy.deepcopy(original)
            if name == 'torch_version':
                report['simulator_cost']['arithmetic_path'][name] = value
            elif name == 'code':
                report['code']['executor_sha256'] = value
            else:
                report[name] = value
            self.write(source/'report.json', report)
            delivery = admission.read(source.parent/'delivery.json')
            delivery['files'] = [self.file_row(source.parent,'source_run/report.json')]
            self.write(source.parent/'delivery.json', delivery)
            with self.subTest(field=name), patch.object(admission, '_source_values',
                    side_effect=AssertionError('scientific arrays opened')):
                with self.assertRaises(ValueError):
                    self.run_metadata(candidate_kind='streamed_cuda')
            self.assertFalse((self.root/'comparison.json').exists())
        self.write(source/'report.json', original)
        delivery['files'] = [self.file_row(source.parent,'source_run/report.json')]
        self.write(source.parent/'delivery.json', delivery)
        with patch.object(admission, '_source_values', side_effect=AssertionError('scientific arrays opened')):
            with self.assertRaisesRegex(ValueError, 'complete nine-case'):
                self.run_metadata(sources=self.sources[:-1], candidate_kind='streamed_cuda')
            with self.assertRaisesRegex(ValueError, 'registered full-scale'):
                self.run_metadata(candidate_kind='streamed_numpy')
        reference_design = admission.read(self.proof/'design.json')
        with self.assertRaisesRegex(ValueError, 'CUDA source comparison requires the full-scale design'):
            comparison._streamed_design_contract(admission, reference_design, self.config,
                sha256_file(self.config_path), None, candidate_backend='cuda')

    def test_full_design_and_distributed_legacy_sources_reach_array_admission(self):
        contract = self.contract()
        self.assertTrue(contract['full_scale_design'])
        self.assertEqual(contract['reference_synapse_execution'], 'factorized_uniform_projections')
        self.assertEqual(contract['reference_execution_implementation_sha256'], sha256_file(self.proof_helper))
        with patch.object(admission, '_source_header', wraps=admission._source_header) as headers, \
             patch.object(admission, '_source_values', side_effect=RuntimeError('SCIENTIFIC_ARRAYS_NOT_LOADED')):
            with self.assertRaisesRegex(RuntimeError, 'SCIENTIFIC_ARRAYS_NOT_LOADED'):
                self.run_metadata()
        self.assertEqual(headers.call_count, 9)
        self.assertFalse((self.root/'comparison.json').exists())

    def test_full_design_rejects_wrong_scale_config_and_runtime_declarations(self):
        for key, value in [('units',2560), ('weight_scaling_factor',1.), ('dt_ms',.025),
                           ('candidate_numpy_version','1.26.4')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.contract(design=dict(self.design, **{key:value}))
        with self.assertRaisesRegex(ValueError, 'config bytes or scale'):
            self.contract(config_sha='wrong-config-bytes')
        with self.assertRaisesRegex(ValueError, 'config bytes or scale'):
            self.contract(config=dict(self.config, excitatory_neurons=2048))

    def test_missing_proof_and_wrong_helper_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires archived factorization proof'):
            self.contract(proof=False)
        with self.assertRaisesRegex(ValueError, 'helper, prior report or parity'):
            self.contract(design=dict(self.design, reference_execution_implementation_sha256='unapproved-helper'))
        self.proof_helper.write_text('# Changed helper engineering fixture\n')
        with self.assertRaisesRegex(ValueError, 'helper, prior report or parity'):
            self.contract()

    def test_changed_or_partial_factorization_proof_is_rejected(self):
        path = self.proof/'source_comparison.json'
        proof = admission.read(path)
        proof['cases'].pop()
        self.write(path, proof)
        with self.assertRaisesRegex(ValueError, 'parity proof changed'):
            self.contract()
        # Even an explicitly rebound engineering receipt cannot admit eight cases.
        changed = dict(self.design, reference_factorization_parity_sha256=sha256_file(path))
        with self.assertRaisesRegex(ValueError, 'complete reference-size factorization parity'):
            self.contract(design=changed)

    def test_native_helper_and_config_must_match_registered_reference(self):
        report_path = self.native/'report.json'
        original = admission.read(report_path)
        changed = copy.deepcopy(original)
        changed['resolved_config']['weight_scaling_factor'] = 1.
        self.write(report_path, changed)
        self.rebind_reference()
        with self.assertRaisesRegex(ValueError, 'reference config differs'):
            self.run_metadata()
        self.write(report_path, original)
        self.native_helper.write_text('# Changed archived native helper\n')
        self.rebind_reference()
        with self.assertRaisesRegex(ValueError, 'helper or execution receipt'):
            self.run_metadata()

    def test_native_runtime_receipt_versions_and_report_hash_are_required(self):
        path = self.archive/'execution.json'
        original = admission.read(path)
        changed = copy.deepcopy(original)
        changed['runtime_versions']['scipy'] = 'different'
        self.write(path, changed)
        self.rebind_reference()
        with self.assertRaisesRegex(ValueError, 'runtime receipt differs'):
            self.run_metadata()
        self.write(path, dict(original, native_report_sha256='unbound-native-report'))
        self.rebind_reference()
        with self.assertRaisesRegex(ValueError, 'helper or execution receipt'):
            self.run_metadata()

    def test_full_design_rejects_merged_batch_and_incomplete_or_duplicate_archives(self):
        with self.assertRaisesRegex(ValueError, 'batch argument is ambiguous'):
            self.run_metadata(candidate=self.root/'synthetic_merged_batch')
        with self.assertRaisesRegex(ValueError, 'complete nine-case'):
            self.run_metadata(sources=self.sources[:-1])
        with patch.object(admission, '_source_values', side_effect=AssertionError('arrays opened')):
            with self.assertRaisesRegex(ValueError, 'duplicate streamed source case'):
                self.run_metadata(sources=[self.sources[0], *self.sources[:-1]])

    def test_raw_validator_supports_explicit_100k_units_and_keeps_default_boundary(self):
        ticks = np.array([0,100000], dtype=np.int64)
        ids = np.array([0,99999], dtype=np.int64)
        path = self.root/'engineering_native.npz'
        np.savez(path, spike_ticks=ticks, spike_unit_indices=ids, spike_seconds=ticks*5e-5,
            native_spike_counts=np.bincount(ids, minlength=100000), terminal_voltage_mV=np.full(100000,-60.))
        case = dict(raw_file=path.name, raw_sha256=sha256_file(path), native_events=2)
        result = admission.raw_values(self.root, case, unit_count=100000)
        self.assertEqual(result['native_spike_counts'].shape, (100000,))
        self.assertEqual(result['spike_unit_indices'][-1], 99999)
        with self.assertRaisesRegex(ValueError, 'native event clock/unit support'):
            admission.raw_values(self.root, case)



class ReferenceClockCorrespondenceTests(unittest.TestCase):
    """Full-neuron metadata and synthetic arrays; no native or learned simulation."""

    write = staticmethod(FullScaleComparisonContractTests.write)
    arrays = staticmethod(StreamedSourceComparisonTests.arrays)
    file_row = staticmethod(FullScaleComparisonContractTests.file_row)

    def setUp(self):
        self.legacy = FullScaleComparisonContractTests()
        self.legacy.setUp()
        self.addCleanup(self.legacy.doCleanups)
        self.root, self.proof = self.legacy.root, self.legacy.proof
        self.config = copy.deepcopy(self.legacy.config)
        self.config['dt_ms'] = .0125
        self.config_path = self.root/'canonical_source.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config,sort_keys=False))
        self.case_ids = ['pfc_94001_cue','pfc_94001_strong_cue']
        self.fixture = source_fixture.SourceReadoutTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.config_path, self.fixture.config = self.config_path, self.config
        _, self.fixture.registered, self.fixture.seeds, self.fixture.dimensions = admission._source_registration(self.config_path)
        self.sources = []
        for case_id in self.case_ids:
            source = self.fixture.archive(case_id)
            StreamedSourceComparisonTests.expand_source(self,source)
            raw = self.arrays(source/'raw.npz')
            raw['spike_ticks'] *= 4
            raw['spike_seconds'] = raw['spike_ticks']*(.0125/1000)
            np.savez(source/'raw.npz',**raw)
            report = admission.read(source/'report.json')
            report['driver'].update(dt_seconds=.0125/1000,total_ticks=400000,ticks_generated=400000,
                binomial_probability=1.4*(.0125/1000),numpy_version='2.4.2')
            report['completed_duration_ticks'] = 400000
            report['inputs_used']['ticks_consumed'] = 400000
            report['inputs_used']['E']['values'] = 80000*400000
            report['inputs_used']['I']['values'] = 20000*400000
            report['simulator_cost']['arithmetic_path']['numpy_version'] = '2.4.2'
            qualification_root = source.parent/'qualification'
            qualifier = admission.read(qualification_root/'comparison/qualification.json')
            qualifier.update(validated_dt_ms=[.0125],numpy_version='2.4.2')
            self.write(qualification_root/'comparison/qualification.json',qualifier)
            self.write(qualification_root/'delivery.json',dict(status='ARCHIVED_PFC_STREAM_VALIDATION',
                result_status=qualifier['status'],files=[self.file_row(qualification_root,'comparison/qualification.json')]))
            report['backend_qualification'].update(qualification=qualifier,
                receipt_sha256=sha256_file(qualification_root/'comparison/qualification.json'),
                delivery_sha256=sha256_file(qualification_root/'delivery.json'))
            self.write(source/'report.json',report)
            request = admission.read(source/'request.json')
            for key in ('case_id','selected_case','registered_case_ids','registered_case_count','config_sha256',
                        'backend','dimensions','code','backend_qualification'):
                request[key] = copy.deepcopy(report[key])
            self.write(source/'request.json',request)
            self.fixture.rebind(source)
            self.sources.append(source)
        self.prepared = self.root/'prepared_reference'
        self.design = comparison.prepare_reference(self.config_path,self.sources,self.proof,self.prepared,self.case_ids)
        self.design_path = self.prepared/'design.json'
        self.archive = self.root/'fine_native_archive'
        self.native = self.archive/'native'
        self.native.mkdir(parents=True)
        shutil.copyfile(self.prepared/'native_config.yaml',self.archive/'native_config.yaml')
        helper = self.archive/'Libs/population_dynamics/pfc_factorized_source.py'
        helper.parent.mkdir(parents=True)
        shutil.copyfile(self.legacy.proof_helper,helper)
        projected = yaml.safe_load((self.prepared/'native_config.yaml').read_text())
        report = dict(admission.read(self.proof/'factorized/report.json'),resolved_config=projected,cases=[])
        for case, source in zip(projected['cases'],self.sources):
            raw = self.native/(case['case_id']+'_native.npz')
            shutil.copyfile(source/'raw.npz',raw)
            report['cases'].append(dict(case,core_units=100000,raw_file=raw.name,raw_sha256=sha256_file(raw),
                elapsed_seconds=2.,native_events=admission.read(source/'report.json')['neural_event_count']))
        self.write(self.native/'report.json',report)
        self.write(self.archive/'execution.json',dict(status='NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE',
            exit_code=0,completed_cases=2,native_report_sha256=sha256_file(self.native/'report.json'),
            design_sha256=sha256_file(self.design_path),native_config_sha256=self.design['native_config_sha256'],
            source_config_sha256=sha256_file(self.config_path),runtime_versions=dict(
                python='engineering fixture',brian2='2.5.4',numpy='1.26.4',scipy='1.12.0')))
        self.rebind_native()

    def rebind_native(self):
        execution = admission.read(self.archive/'execution.json')
        execution['native_report_sha256'] = sha256_file(self.native/'report.json')
        self.write(self.archive/'execution.json',execution)
        self.write(self.archive/'delivery.json',dict(files=[self.file_row(self.archive,name) for name in
            ('native/report.json','execution.json','native_config.yaml','Libs/population_dynamics/pfc_factorized_source.py')]))

    def compare(self,name='fine_comparison.json',sources=None):
        return comparison.compare(None,[self.native],self.design_path,self.root/name,
            candidate_kind='streamed_numpy',config_path=self.config_path,
            source_runs=self.sources if sources is None else sources,factorization_proof=self.proof)

    def test_prepare_reference_real_entry_projects_only_cases_without_source_arrays(self):
        output = self.root/'cli_prepared'
        original = self.config_path.read_bytes()
        arguments = ['prepare-reference','--config',str(self.config_path),'--factorization-proof',str(self.proof),
                     '--output',str(output)]
        for source in self.sources:arguments += ['--source-run',str(source)]
        for case_id in self.case_ids:arguments += ['--case-id',case_id]
        with patch.object(admission,'_source_values',side_effect=AssertionError('source arrays read')):
            self.assertEqual(comparison.main(arguments),0)
        projected = yaml.safe_load((output/'native_config.yaml').read_text())
        expected = copy.deepcopy(self.config)
        expected['cases'] = [case for case in expected['cases'] if case['case_id'] in self.case_ids]
        self.assertEqual(projected,expected)
        self.assertEqual(self.config_path.read_bytes(),original)
        design = admission.read(output/'design.json')
        self.assertEqual(design['native_config_sha256'],sha256_file(output/'native_config.yaml'))
        self.assertEqual(design['source_config_sha256'],sha256_file(self.config_path))
        self.assertEqual(design['canonical_registered_cases'],9)
        self.assertEqual(design['case_ids'],self.case_ids)
        self.assertEqual(design['scope'],comparison.REFERENCE_CLOCK_SCOPE)
        self.assertFalse(design['full_scale_qualified'])
        self.assertFalse(design['working_memory_qualified'])

    def test_complete_full_neuron_correspondence_keeps_source_canonical_denominator(self):
        with patch.object(admission,'_source_values',wraps=admission._source_values) as values:
            report = self.compare()
        self.assertEqual(report['status'],'PASS_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE')
        self.assertEqual(report['registered_cases'],2)
        self.assertEqual(report['compared_cases'],2)
        self.assertEqual(report['canonical_registered_cases'],9)
        self.assertEqual(len(report['uncompared_canonical_case_ids']),7)
        self.assertEqual(report['source_rng_seeds'],[94001])
        self.assertTrue(all(len(call.args[2])==9 for call in values.call_args_list))
        self.assertEqual(report['native_config_sha256'],self.design['native_config_sha256'])
        for flag in ('full_scale_qualified','scientific_fidelity_qualified','working_memory_qualified',
                     'fidelity_at_requested_scale_qualified'):
            self.assertFalse(report[flag])
        self.assertEqual(report['production_episodes'],0)

    def test_single_preexisting_canonical_case_is_explicitly_preregistered(self):
        output = self.root/'single_prepared'
        design = comparison.prepare_reference(self.config_path,[self.sources[1]],self.proof,output,[self.case_ids[1]])
        self.assertEqual(design['cases'],1)
        self.assertEqual(design['case_ids'],['pfc_94001_strong_cue'])
        self.assertEqual(len(design['canonical_case_ids']),9)
        self.assertEqual(design['preexisting_source_bindings'][0]['report_sha256'],sha256_file(self.sources[1]/'report.json'))

    def test_missing_duplicate_wrong_case_and_unregistered_selection_reject_before_preparation(self):
        for sources,cases in (([],self.case_ids),(self.sources,[self.case_ids[0],self.case_ids[0]]),
            ([self.sources[0]],[self.case_ids[1]]),(self.sources,['unregistered'])):
            output = self.root/'rejected_prepare'
            with self.subTest(cases=cases),self.assertRaises(ValueError):
                comparison.prepare_reference(self.config_path,sources,self.proof,output,cases)
            self.assertFalse(output.exists())

    def test_preparation_rejects_incomplete_full_neuron_metadata(self):
        source = self.sources[0]
        report = admission.read(source/'report.json')
        report['dimensions']['n_e'] = 16
        self.write(source/'report.json',report);self.fixture.rebind(source)
        output = self.root/'bad_dimensions'
        with self.assertRaisesRegex(ValueError,'source execution/configuration binding'):
            comparison.prepare_reference(self.config_path,self.sources,self.proof,output,self.case_ids)
        self.assertFalse(output.exists())

    def test_native_case_projection_rejects_physical_clock_rng_and_case_drift_before_arrays(self):
        path = self.native/'report.json'
        original = admission.read(path)
        for changes in (dict(dt_ms=.025),dict(weight_scaling_factor=1.),dict(stimulus_width_deg=40.),dict(source_rng_seed=1)):
            changed = copy.deepcopy(original);changed['resolved_config'].update(changes)
            self.write(path,changed);self.rebind_native()
            with self.subTest(changes=changes),patch.object(admission,'_source_values',side_effect=AssertionError('arrays read')):
                with self.assertRaisesRegex(ValueError,'reference config differs'):
                    self.compare()
        changed = copy.deepcopy(original);changed['cases'][0]['source_rng_seed'] = 94002
        self.write(path,changed);self.rebind_native()
        with self.assertRaisesRegex(ValueError,'seed, control or unit identity'):
            self.compare()

    def test_native_projection_file_hash_and_rebound_noncase_changes_reject(self):
        path = self.prepared/'native_config.yaml'
        projected = yaml.safe_load(path.read_text());projected['stimulus_width_deg'] = 40.
        path.write_text(yaml.safe_dump(projected))
        with self.assertRaisesRegex(ValueError,'projection or hash changed'):
            self.compare()
        changed = dict(self.design,native_config_sha256=sha256_file(path))
        self.write(self.design_path,changed)
        with self.assertRaisesRegex(ValueError,'projection or hash changed'):
            self.compare()

    def test_wrong_native_snapshot_design_or_runtime_rejects_before_arrays(self):
        path = self.archive/'execution.json'
        original = admission.read(path)
        for field,value in (('design_sha256','0'*64),('source_config_sha256','0'*64),('native_config_sha256','0'*64)):
            self.write(path,dict(original,**{field:value}));self.rebind_native()
            with self.subTest(field=field),patch.object(admission,'_source_values',side_effect=AssertionError('arrays read')):
                with self.assertRaisesRegex(ValueError,'preregistration or configuration binding'):
                    self.compare()
        changed = copy.deepcopy(original);changed['runtime_versions']['scipy'] = 'other'
        self.write(path,changed);self.rebind_native()
        with self.assertRaisesRegex(ValueError,'runtime receipt differs'):
            self.compare()

    def test_changed_proof_helper_and_candidate_runtime_are_not_rebound_away(self):
        helper = self.archive/'Libs/population_dynamics/pfc_factorized_source.py'
        old = helper.read_bytes();helper.write_text('# changed helper\n');self.rebind_native()
        with self.assertRaisesRegex(ValueError,'helper or execution receipt'):
            self.compare()
        helper.write_bytes(old);self.rebind_native()
        source = self.sources[0];report = admission.read(source/'report.json')
        report['driver']['numpy_version'] = '1.26.4'
        self.write(source/'report.json',report);self.fixture.rebind(source)
        design = copy.deepcopy(self.design)
        design['preexisting_source_bindings'][0].update(report_sha256=sha256_file(source/'report.json'),
            delivery_sha256=sha256_file(source.parent/'delivery.json'))
        self.write(self.design_path,design)
        execution = admission.read(self.archive/'execution.json')
        execution['design_sha256'] = sha256_file(self.design_path)
        self.write(self.archive/'execution.json',execution);self.rebind_native()
        with patch.object(admission,'_source_values',side_effect=AssertionError('arrays read')):
            with self.assertRaisesRegex(ValueError,'candidate runtime changed'):
                self.compare()

    def test_selected_source_omissions_duplicates_and_post_registration_changes_reject(self):
        for sources in (self.sources[:1],[self.sources[0],self.sources[0]]):
            with self.subTest(count=len(sources)),self.assertRaises(ValueError):
                self.compare(sources=sources)
        source = self.sources[0];report = admission.read(source/'report.json')
        report['selected_case']['stimulus_strength_nA'] = .4
        self.write(source/'report.json',report);self.fixture.rebind(source)
        with self.assertRaises(ValueError):self.compare()

    def test_numeric_failures_retain_entire_subset_and_exit_nonzero(self):
        native_report = admission.read(self.native/'report.json')
        for case in native_report['cases']:
            path = self.native/case['raw_file'];arrays = self.arrays(path)
            arrays['terminal_voltage_mV'] += 2e-10
            np.savez(path,**arrays);case['raw_sha256'] = sha256_file(path)
        self.write(self.native/'report.json',native_report);self.rebind_native()
        output = self.root/'numeric_failure.json'
        arguments = ['--candidate-kind','streamed_numpy','--reference',str(self.native),'--design',str(self.design_path),
            '--config',str(self.config_path),'--factorization-proof',str(self.proof),'--output',str(output)]
        for source in self.sources:arguments += ['--source-run',str(source)]
        self.assertEqual(comparison.main(arguments),1)
        result = admission.read(output)
        self.assertEqual(result['status'],'FAILED_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE')
        self.assertEqual(result['compared_cases'],2)
        self.assertEqual(result['failed_case_ids'],self.case_ids)
        self.assertEqual(result['passed_cases'],0)
        self.assertTrue(all(row['failed_arrays']==['terminal_voltage_mV'] for row in result['cases']))
        self.assertFalse(result['scientific_fidelity_qualified'])

    def test_explicit_fine_raw_clock_and_duration_preserve_legacy_rejection(self):
        path = self.root/'fine_raw.npz'
        ticks = np.array([0,400000],dtype=np.int64);ids = np.array([0,99999],dtype=np.int64)
        np.savez(path,spike_ticks=ticks,spike_unit_indices=ids,spike_seconds=ticks*.0125/1000,
            native_spike_counts=np.bincount(ids,minlength=100000),terminal_voltage_mV=np.full(100000,-60.))
        case = dict(raw_file=path.name,raw_sha256=sha256_file(path),native_events=2)
        with self.assertRaisesRegex(ValueError,'event clock/unit support'):
            admission.raw_values(self.root,case,unit_count=100000)
        values = admission.raw_values(self.root,case,unit_count=100000,dt_ms=.0125,duration_ticks=400000)
        self.assertEqual(values['spike_ticks'][-1],400000)
        with self.assertRaisesRegex(ValueError,'seconds/ticks differ'):
            admission.raw_values(self.root,case,unit_count=100000,dt_ms=.025,duration_ticks=400000)
        with self.assertRaisesRegex(ValueError,'event clock/unit support'):
            admission.raw_values(self.root,case,unit_count=100000,dt_ms=.0125,duration_ticks=100000)

    def test_legacy_full_scale_scope_cannot_admit_fine_clock_or_subset(self):
        design = dict(self.design,scope='native_full_scale_source_equivalence')
        with self.assertRaisesRegex(ValueError,'registered full-scale'):
            comparison._streamed_design_contract(admission,design,self.config,sha256_file(self.config_path),self.proof)


if __name__ == '__main__':
    unittest.main()
