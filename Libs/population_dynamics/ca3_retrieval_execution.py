"""Run the pinned CA3 retrieval using its independently verified normalized65p.

By default only the author's absolute input path is adapted. The explicit
cue-omission development condition retains ESTDP, RNG draws and the1001ms clock
while omitting cue overwrites. Neither condition establishes pattern completion
or an exact continuation of formation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

from .ca3_native_observation import ENVIRONMENT_VARIABLE, PINNED_FILES, summarize
from .carlsim_records import SOURCE_REVISION, network_header, validate_coupled_identity, validate_retrieval
from .provenance import sha256_file
from .resource_probe import probe

AUTHOR_REVISION = '1989764942516bf110dafc29ad08d815fd699f4c'
INPUT_NAME = 'network65p_assem275_divnorm.dat'


def read(path):
    return json.loads(Path(path).read_text())


def observation_environment(runtime: Path, report: dict, *, enabled: bool) -> dict:
    """Bind opt-in recording to the observed build and a fresh runtime output."""
    overlay = report.get('native_observation_overlay')
    if ENVIRONMENT_VARIABLE in os.environ:
        raise ValueError('CA3_OBSERVATION_REQUIRES_EXPLICIT_RUNNER_BINDING')
    if bool(overlay) != enabled:
        raise ValueError('CA3_OBSERVATION_BUILD_AND_EXECUTION_FLAGS_DIFFER')
    if not enabled:
        return {}
    if (overlay.get('enabled_by') != ENVIRONMENT_VARIABLE
            or overlay.get('framework_revision') != SOURCE_REVISION
            or overlay.get('original_sha256') != PINNED_FILES
            or overlay.get('native_build_verified') is not True
            or report.get('observer_implementation_sha256') != sha256_file(
                Path(__file__).with_name('ca3_native_observation.py'))):
        raise ValueError('CA3_VERIFIED_OBSERVATION_BUILD_REQUIRED')
    trace = runtime/'native_observation.bin'
    if trace.exists() or trace.is_symlink():
        raise FileExistsError(trace)
    return {ENVIRONMENT_VARIABLE: str(trace)}


def qualify_input(formation: Path, formation_replay: Path, normalized: Path,
                  normalization: Path, normalization_replay: Path, *, coupled_extension: bool = False,
                  source_realization=None) -> dict:
    """Bind the completed formation and both independent replays to actual input."""
    native_path = formation/'native_report.json'
    native, replay = read(native_path), read(formation_replay)
    normal, checked = read(normalization), read(normalization_replay)
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    if coupled_extension:
        validate_coupled_identity(native)
    if (native['status'] != family+'_FORMATION_EXIT_ZERO' or native['exit_code'] != 0
            or native['expected_duration_ms'] != 70000 or native['core_neurons'] != (168106 if coupled_extension else 84053)
            or native['author_revision'] != AUTHOR_REVISION or native['framework_revision'] != SOURCE_REVISION
            or native.get('complete_author_formation_program_unchanged') is not (not coupled_extension)
            or native.get('complete_formation_executed') is not True
            or replay['status'] != 'PASS_'+family+'_FORMATION_PREFIX_REPLAY_AND_COUNTS'
            or replay['formation_report_sha256'] != sha256_file(native_path)
            or replay['formation_recording_sha256'] != sha256_file(formation/'observations/recording.json')):
        raise ValueError('CA3_COMPLETE_REPLAYED_FORMATION_REQUIRED')
    if coupled_extension and replay.get('coupled_extension') != native['coupled_extension']:
        raise ValueError('CA3_FORMATION_REPLAY_COUPLED_IDENTITY_DIFFERS')
    source = formation/'native_run/network65p.dat'
    raw_sha, normalized_sha = sha256_file(source), sha256_file(normalized)
    if (normal['status'] != 'APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION'
            or checked['status'] != 'PASS_NORMALIZATION_REPLAY'
            or checked['all_record_bytes_verified'] is not True
            or checked['normalization_report_sha256'] != sha256_file(normalization)
            or checked['source_sha256'] != normal['source_sha256'] or normal['source_sha256'] != raw_sha
            or checked['normalized_sha256'] != normal['output_sha256'] or normal['output_sha256'] != normalized_sha
            or checked['weight_tolerance'] != 0 or checked['target_global_mean'] != 0.625):
        raise ValueError('CA3_COMPLETE_AUTHOR_NORMALIZATION_REPLAY_REQUIRED')
    header = network_header(normalized)
    # Float32 serialization of the author's65p time, 45.85 seconds.
    if (header['simulation_seconds'] != 45.849998474121094 or header['units'] != native['core_neurons']
            or {g.name:g.units for g in header['groups']} != native['populations']):
        raise ValueError('CA3_NORMALIZED65P_IDENTITY_REQUIRED')
    binding = dict(normalized_input_qualified=True, normalized_input_sha256=normalized_sha,
                raw65p_sha256=raw_sha, formation_report_sha256=sha256_file(native_path),
                formation_replay_sha256=sha256_file(formation_replay),
                normalization_report_sha256=sha256_file(normalization),
                normalization_replay_sha256=sha256_file(normalization_replay),
                populations=native['populations'], pattern_sha256=native['pattern_sha256'])
    if coupled_extension:
        binding['coupled_extension_config_sha256'] = native['coupled_extension_config']['sha256']
    if source_realization is not None or 'source_realization' in native:
        from .ca3_realization import bind_circuit_input

        binding['circuit_identity'] = bind_circuit_input(native, replay, source_realization, normalized_sha)
    return binding


def execute(runtime: Path, formation: Path, formation_replay: Path, normalized: Path,
             normalization: Path, normalization_replay: Path, *, native_observation: bool = False,
            coupled_extension: bool = False, cue_omission: bool = False,
            neuron_substeps: int | None = None, source_realization_config: Path | None = None) -> dict:
    runtime = runtime.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('CA3_RETRIEVAL_REQUIRES_ALLOCATED_GPU')
    report_path = runtime/'native_report.json'
    if report_path.exists() or (runtime/'native_run').exists():
        raise FileExistsError('CA3_RETRIEVAL_OUTPUT_ALREADY_EXISTS')
    for row in read(runtime/'manifest.json')['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('CA3_RUNTIME_PATH_INVALID')
        if sha256_file(runtime/relative) != row['sha256']:
            raise ValueError('CA3_RUNTIME_INPUT_CHANGED:'+str(relative))
    report = read(runtime/'build_report.json')
    realization = None
    if source_realization_config is not None or 'source_realization' in report:
        from .ca3_realization import bind_runtime

        realization = bind_runtime(runtime, report, source_realization_config)
        if not coupled_extension or cue_omission or neuron_substeps is not None:
            raise ValueError('CA3_REALIZATION_SOURCE_CLOCK_RETRIEVAL_REQUIRED')
    if neuron_substeps is not None and (not coupled_extension or not native_observation or cue_omission):
        raise ValueError('CA3_REFINEMENT_REQUIRES_OBSERVED_COUPLED_SOURCE_RETRIEVAL')
    condition = 'cue_omission' if cue_omission else 'source'
    if type(cue_omission) is not bool or report.get('retrieval_condition', 'source') != condition:
        raise ValueError('CA3_RETRIEVAL_CONDITION_FLAGS_DIFFER')
    if cue_omission:
        from .ca3_cue_omission import validate_report

        if not native_observation:
            raise ValueError('CA3_OMISSION_REQUIRES_NATIVE_OBSERVATION')
        validate_report(report, population_multiplier=2 if coupled_extension else 1)
    observer_env = observation_environment(runtime, report, enabled=native_observation)
    contract = report['retrieval_input_contract']
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    input_name = 'network65p_assem550_divnorm.dat' if coupled_extension else INPUT_NAME
    if coupled_extension:
        validate_coupled_identity(report, neuron_substeps=neuron_substeps)
    expected_build = ('COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED'
                      if coupled_extension else 'NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING')
    if realization is not None:
        expected_build = 'CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING'
    expected_scope = 'coupled_extension_retrieval' if coupled_extension else 'native_source_retrieval'
    binary = runtime/'bin/ca3_retrieval'
    if (report['status'] != expected_build
            or report['scope'] != expected_scope or report['expected_duration_ms'] != 1001
            or report['author_revision'] != AUTHOR_REVISION or report['framework_revision'] != SOURCE_REVISION
            or report['retrieval_program_changed_only_in_input_path'] is not (not coupled_extension and not cue_omission)
            or contract['effective_input_binding'] != input_name or contract['source_path_binding_only'] is not (not coupled_extension and not cue_omission)
            or (coupled_extension and contract.get('fresh_coupled_checkpoint_required') is not True)
            or contract['source_carlsim_seed'] != 10 or contract['source_host_srand_seed'] != 3
            or contract['source_estdp_enabled'] is not True
            or sha256_file(binary) != report['binary_sha256']):
        raise ValueError('CA3_PATH_ONLY_RETRIEVAL_BUILD_REQUIRED')
    binding = qualify_input(formation, formation_replay, normalized, normalization, normalization_replay,
                            coupled_extension=coupled_extension, source_realization=realization)
    if realization is not None:
        report['realization_circuit_binding'] = dict(binding['circuit_identity'],
            normalized_input_sha256=binding['normalized_input_sha256'],
            formation_report_sha256=binding['formation_report_sha256'],
            block_id=realization.block_id, new_circuit_claimed=False,
            microscopic_state_restarted_from_checkpoint=False)
    if coupled_extension and binding['coupled_extension_config_sha256'] != report['coupled_extension_config']['sha256']:
        raise ValueError('CA3_FORMATION_RETRIEVAL_COUPLED_CONFIG_DIFFERS')
    if (report['populations'] != binding['populations'] or report['pattern_sha256'] != binding['pattern_sha256']
            or sha256_file(runtime/'trainPat.csv') != binding['pattern_sha256']):
        raise ValueError('CA3_FORMATION_RETRIEVAL_PATTERN_OR_POPULATION_DIFFERS')
    report.update(binding, status='PREFLIGHT', native_run_executed=False, retrieval_executed=False,
                  complete_formation_retrieval=False, pattern_completion_qualified=False,
                  slurm_job_id=os.environ['SLURM_JOB_ID'], started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime_manifest_sha256=sha256_file(runtime/'manifest.json'))

    def save():
        temporary = report_path.with_suffix('.writing')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(report_path)

    started = time.perf_counter()
    child = None
    save()
    try:
        report['resource_admission'] = dict(minimum_memory_gib=128 if coupled_extension else 64,
                                            extrapolated=coupled_extension)
        health = probe(runtime/'resource_preflight.json', required_memory_gib=128 if coupled_extension else 64)
        if health['status'] != 'PASS':
            raise RuntimeError('CA3_RETRIEVAL_RESOURCE_PREFLIGHT_FAILED')
        run = runtime/'native_run'
        run.mkdir()
        shutil.copy2(runtime/'trainPat.csv', run/'trainPat.csv')
        (run/input_name).symlink_to(normalized.resolve())
        (runtime/'tmp').mkdir(exist_ok=True)
        env = {**os.environ, **observer_env, 'LD_LIBRARY_PATH':str(runtime/'lib')+(
            ':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
            'TMPDIR':str(runtime/'tmp')}
        native_command = [str(binary)]
        library_command = ['ldd', str(binary)]
        if cue_omission or neuron_substeps is not None or realization is not None:
            # This paired intervention must load the byte-identical reference
            # backend, even while another build's embedded RPATH still exists.
            from .ca3_formation_execution import native_launch_command

            native_command = native_launch_command(runtime, binary)
            library_command = native_command[:-1]+['--list', native_command[-1]]
        libraries = subprocess.check_output(library_command, env=env, text=True)
        if 'not found' in libraries:
            raise RuntimeError(libraries)
        if (cue_omission or neuron_substeps is not None or realization is not None) and any(str(runtime/'lib'/name) not in libraries for name in
                ('libcarlsim.so.6.0.0','libcarlsim-stopwatch.so','libcurand.so.10')):
            raise ValueError('CA3_OMISSION_PACKAGED_BACKEND_NOT_RESOLVED')
        report.update(status='RUNNING_'+family+'_RETRIEVAL', runtime_linked_libraries=libraries,
                       cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        if cue_omission or neuron_substeps is not None or realization is not None:
            report['native_command'] = native_command
        save()
        stop_reason = None
        with (runtime/'native.log').open('x') as log:
            child = subprocess.Popen(native_command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
            report['native_pid'] = child.pid
            save()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if stop_reason is not None:
                        continue
                    health = probe(runtime/'resource_latest.json', required_memory_gib=8)
                    if child.poll() is None and health['status'] != 'PASS':
                        stop_reason = health['reasons']
                        child.terminate()
                        report.update(status='RESOURCE_STOP_REQUESTED', resource_stop_reasons=stop_reason)
                        save()
        report.update(native_run_executed=True, exit_code=child.returncode)
        if child.returncode != 0 or stop_reason is not None:
            raise RuntimeError('CA3_NATIVE_RETRIEVAL_EXIT:'+str(child.returncode))
        if sha256_file(normalized) != binding['normalized_input_sha256']:
            raise ValueError('CA3_RETRIEVAL_INPUT_CHANGED_DURING_EXECUTION')
        if native_observation:
            trace = runtime/'native_observation.bin'
            summary = summarize(trace, expected_stop_ms=1001)
            if ({group.name:group.units for group in summary['groups']} != binding['populations']
                    or summary['unit_counts'].shape != (report['core_neurons'],)):
                raise ValueError('CA3_NATIVE_OBSERVATION_UNIT_IDENTITY_DIFFERS')
            report.update(native_observation_stream_validated=True,
                          native_observation_sha256=sha256_file(trace),
                          native_observation_bytes=trace.stat().st_size,
                          observed_native_duration_ms=summary['duration_ms'],
                          host_request_runtime_current_equal=summary['host_request_runtime_current_equal'],
                          native_counter_event_reconciliation_completed=False)
            report['native_observation_overlay']['native_execution_verified'] = True
        report.update(status=family+'_RETRIEVAL_EXIT_ZERO', retrieval_executed=True,
                      expected_duration_ms=1001, full_state_restart_available=False)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        if child is not None:
            report.update(native_run_executed=True, exit_code=child.returncode)
        raise
    finally:
        report.update(native_elapsed_seconds=time.perf_counter()-started,
                      child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    decode_options = dict(coupled_extension=coupled_extension)
    if cue_omission:
        decode_options['cue_omission'] = True
    if neuron_substeps is not None:
        decode_options['neuron_substeps'] = neuron_substeps
    if realization is not None:
        decode_options['source_realization_config'] = source_realization_config
    validate_retrieval(runtime, runtime/'observations', **decode_options)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'formation', 'formation-replay', 'normalized', 'normalization', 'normalization-replay'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--native-observation', action='store_true',
                        help='Record all per-ms native counters/effective currents from an explicitly observed build')
    parser.add_argument('--coupled-extension', action='store_true',
                        help='Require a fresh168106-unit formation and normalized checkpoint')
    parser.add_argument('--cue-omission', action='store_true')
    parser.add_argument('--neuron-substeps', type=int, choices=(5, 10, 20, 40))
    parser.add_argument('--source-realization-config', type=Path)
    args = parser.parse_args()
    report = execute(args.runtime, args.formation, args.formation_replay, args.normalized,
                     args.normalization, args.normalization_replay, native_observation=args.native_observation,
                     coupled_extension=args.coupled_extension, cue_omission=args.cue_omission,
                     neuron_substeps=args.neuron_substeps, source_realization_config=args.source_realization_config)
    print(json.dumps({k:report[k] for k in ('status', 'retrieval_executed', 'scientific_fidelity_qualified')}))


if __name__ == '__main__':
    main()
