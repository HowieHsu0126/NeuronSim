"""Execute a verified CA3 formation or its independent prefix on an allocated GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import re
import shutil
import subprocess
import time

from .ca3_reference import FORMATION_DURATION_MS, FORMATION_SCOPES, PREFIX_DURATION_MS, verify_checkpoints
from .carlsim_records import validate_coupled_identity, validate_formation, validate_prefix
from .provenance import sha256_file
from .resource_probe import probe


# A prefix retains one checkpoint. The reference's measured native-process peak
# plus that checkpoint is about40GiB;64GiB per reference population provides
# initial headroom. The derived128GiB bound remains an unmeasured extrapolation.
# Full formation retains its original170GiB-per-reference admission.
REFERENCE_PREFIX_MEMORY_GIB = 64


def native_launch_command(runtime: Path, binary: Path) -> list[str]:
    """Keep the compiled bytes while resolving dependencies from the runtime."""
    headers = subprocess.check_output(
        ['readelf', '-l', str(binary)], text=True,
        env={**os.environ, 'LC_ALL': 'C'})
    interpreters = re.findall(r'\[Requesting program interpreter: ([^\]]+)\]', headers)
    if len(interpreters) != 1 or not Path(interpreters[0]).is_absolute():
        raise ValueError('CA3_ELF_INTERPRETER_REQUIRED')
    return [interpreters[0], '--inhibit-rpath', '', '--library-path',
            str(runtime/'lib'), str(binary)]


def native_prefix_author_program_preserved(report: dict) -> bool:
    """Accept the declared prefix stop separately from the known backend correction.

    Historical unmodified-backend reports use the whole-source flag. A changed
    backend requires the newer author-only proof and its explicit correction;
    this does not establish trajectory equivalence or native qualification.
    """
    global_preserved = report.get('source_code_semantics_preserved_before_stop')
    if report.get('author_program_semantics_preserved_before_stop', global_preserved) is not True:
        return False
    memory = report.get('conductance_memory_overlay')
    if global_preserved is True:
        return memory is None and report.get('native_observation_overlay') is None
    if (global_preserved is not False or not isinstance(memory, dict)
            or report.get('native_observation_overlay') is not None):
        return False
    # Frozen unmodified-backend bundles do not require this optional helper.
    from .ca3_conductance_memory import GPU_FILE, OVERLAY_ID, PINNED_SHA256

    correction = memory.get('initialization_correction')
    output = memory.get('overlay_sha256')
    return (memory.get('overlay_id') == OVERLAY_ID
            and memory.get('original_sha256') == {GPU_FILE: PINNED_SHA256}
            and memory.get('input_sha256') == {GPU_FILE: PINNED_SHA256}
            and isinstance(output, dict) and set(output) == {GPU_FILE}
            and isinstance(output[GPU_FILE], str)
            and re.fullmatch(r'[0-9a-f]{64}', output[GPU_FILE]) is not None
            and output[GPU_FILE] != PINNED_SHA256
            and isinstance(correction, dict)
            and correction.get('corrected_extent') == 'syn_gPitch * numNReg'
            and correction.get('previously_uninitialized_rows_are_now_zeroed') is True
            and memory.get('kernel_arithmetic_unchanged') is True
            and memory.get('precision_unchanged') is True
            and report.get('conductance_memory_implementation_sha256') == sha256_file(
                Path(__file__).with_name('ca3_conductance_memory.py')))


def execute(runtime: Path, *, coupled_extension: bool = False,
            scope: str = "full-formation", source_realization_config: Path | None = None) -> dict:
    if scope not in FORMATION_SCOPES:
        raise ValueError("CA3_UNSUPPORTED_FORMATION_EXECUTION_SCOPE")
    prefix = scope == "prefix"
    phase = "PREFIX" if prefix else "FORMATION"
    runtime = Path(runtime).resolve()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('CA3_FORMATION_REQUIRES_ALLOCATED_GPU')
    report_path = runtime/'native_report.json'
    if report_path.exists() or (runtime/'native_run').exists():
        raise FileExistsError('CA3_FORMATION_OUTPUT_ALREADY_EXISTS')
    manifest = json.loads((runtime/'manifest.json').read_text())
    for row in manifest['files']:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('CA3_RUNTIME_PATH_INVALID')
        if sha256_file(runtime/path) != row['sha256']:
            raise ValueError('CA3_RUNTIME_INPUT_CHANGED:' + str(path))
    report = json.loads((runtime/'build_report.json').read_text())
    realization = None
    if source_realization_config is not None or 'source_realization' in report:
        from .ca3_realization import bind_runtime

        if not coupled_extension:
            raise ValueError('CA3_REALIZATION_COUPLED_FORMATION_REQUIRED')
        realization = bind_runtime(runtime, report, source_realization_config)
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    if coupled_extension:
        validate_coupled_identity(report)
    expected_scope = ('coupled_extension_' if coupled_extension else 'native_source_') + ('prefix' if prefix else 'formation')
    expected_build = ('COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED'
                      if coupled_extension else 'NATIVE_SOURCE_BUILD_PASSED_GPU_RUN_PENDING')
    if realization is not None:
        expected_build = 'CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING'
    if coupled_extension or prefix:
        source_match = (report['complete_author_formation_program_unchanged'] is False
                        and report['overlay_source_sha256'] != report['original_author_main_sha256'])
        if not coupled_extension:
            source_match = source_match and native_prefix_author_program_preserved(report)
    else:
        source_match = (report['complete_author_formation_program_unchanged'] is True
                        and report['overlay_source_sha256'] == report['original_author_main_sha256'])
    expected_duration = PREFIX_DURATION_MS if prefix else FORMATION_DURATION_MS
    if (report['scope'] != expected_scope or report['status'] != expected_build
            or not source_match or report['expected_duration_ms'] != expected_duration
            or report['core_neurons'] != (168106 if coupled_extension else 84053)):
        if prefix:
            raise ValueError('CA3_COUPLED_PREFIX_BUILD_REQUIRED' if coupled_extension
                             else 'CA3_NATIVE_PREFIX_BUILD_REQUIRED')
        raise ValueError('CA3_COUPLED_FULL_FORMATION_BUILD_REQUIRED' if coupled_extension
                         else 'CA3_UNCHANGED_FULL_FORMATION_BUILD_REQUIRED')
    binary = runtime/'bin'/('ca3_prefix' if prefix else 'ca3_formation')
    if sha256_file(binary) != report['binary_sha256']:
        raise ValueError('CA3_FORMATION_BINARY_DIFFERS_FROM_BUILD')
    report.update(status='PREFLIGHT', slurm_job_id=os.environ['SLURM_JOB_ID'],
                  started_utc=datetime.now(timezone.utc).isoformat(), native_run_executed=False,
                  runtime_manifest_sha256=sha256_file(runtime/'manifest.json'),
                  complete_formation_executed=False, complete_formation_retrieval=False,
                  full_state_restart_available=False)

    if realization is not None:
        report['realization_circuit_binding'] = dict(circuit_id=realization.circuit_id,
            circuit_seed=realization.circuit_seed, source_realization_config_sha256=sha256_file(source_realization_config),
            fresh_native_graph_constructed=False, independent_circuits_qualified=False,
            microscopic_state_restarted_from_checkpoint=False)

    def save():
        temporary = report_path.with_suffix('.writing')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(report_path)

    started = time.perf_counter()
    save()
    child = None
    try:
        scale = 2 if coupled_extension else 1
        # Initial constant-expected-degree allocation estimate, not a measured
        # resource qualification for the derived model.
        minimum_memory = (REFERENCE_PREFIX_MEMORY_GIB if prefix else 170) * scale
        report['resource_admission'] = dict(minimum_memory_gib=minimum_memory,
            reference_population_multiplier=scale, extrapolated=coupled_extension)
        if prefix:
            report['resource_admission'].update(retained_checkpoints=1,
                planning_basis='reference native-process peak plus one checkpoint and headroom')
        initial = probe(runtime/'resource_preflight.json', required_memory_gib=minimum_memory)
        required_scratch = (1 if prefix else 20) * manifest['reference_checkpoint_bytes'] * scale + 8 * 1024**3
        if initial['status'] != 'PASS' or shutil.disk_usage(runtime).free < required_scratch:
            raise RuntimeError('CA3_FULL_FORMATION_RESOURCE_PREFLIGHT_FAILED')
        run = runtime/'native_run'
        run.mkdir()
        shutil.copy2(runtime/'trainPat.csv', run/'trainPat.csv')
        temporary = runtime/'tmp'
        temporary.mkdir(exist_ok=True)
        env = {**os.environ, 'LD_LIBRARY_PATH': str(runtime/'lib') + (
            ':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
            'TMPDIR': str(temporary), 'OMP_NUM_THREADS': os.environ.get('SLURM_CPUS_PER_TASK', '1')}
        command = native_launch_command(runtime, binary) if prefix or realization is not None else [str(binary)]
        link_command = command[:-1]+['--list',command[-1]] if prefix or realization is not None else ['ldd',str(binary)]
        linked = subprocess.check_output(link_command, env=env, text=True)
        if 'not found' in linked:
            raise RuntimeError(linked)
        if realization is not None and any(str(runtime/'lib'/name) not in linked for name in
                ('libcarlsim.so.6.0.0','libcarlsim-stopwatch.so','libcurand.so.10')):
            raise ValueError('CA3_REALIZATION_PACKAGED_BACKEND_NOT_RESOLVED')
        report.update(status='RUNNING_'+family+'_'+phase, runtime_linked_libraries=linked,
                      native_launch_command=command, runtime_library_check_command=link_command,
                      native_run_directory=str(run), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        save()
        stop_reason = None
        report['preflight_elapsed_seconds'] = time.perf_counter() - started
        process_started = time.perf_counter()
        with (runtime/'native.log').open('x') as log:
            child = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
            report['native_pid'] = child.pid
            save()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if stop_reason is not None:
                        continue  # A blocked child may take time to terminate; never send SIGKILL.
                    health = probe(runtime/'resource_latest.json', required_memory_gib=8)
                    if child.poll() is None and health['status'] != 'PASS':
                        stop_reason = health['reasons']
                        child.terminate()
                        report.update(status='RESOURCE_STOP_REQUESTED', resource_stop_reasons=stop_reason)
                        save()
        report.update(native_run_executed=True, exit_code=child.returncode,
                      native_process_elapsed_seconds=time.perf_counter()-process_started)
        if child.returncode != 0 or stop_reason is not None:
            raise RuntimeError('CA3_NATIVE_'+phase+'_EXIT:' + str(child.returncode))
        checkpoints = verify_checkpoints(run, scope, expected_units=report['core_neurons'])
        checkpoint = run/('network5p.dat' if prefix else 'network100p.dat')
        report.update(status=family+'_'+phase+'_EXIT_ZERO', complete_formation_executed=not prefix,
                      checkpoints=checkpoints, checkpoint_bytes=checkpoint.stat().st_size,
                      checkpoint_sha256=sha256_file(checkpoint),
                      checkpoint_relative_path='native_run/'+checkpoint.name)
        if realization is not None:
            report['realization_circuit_binding'].update(fresh_native_graph_constructed=True,
                formation_checkpoint_sha256=report['checkpoint_sha256'])
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()  # Preserve the actual child outcome; do not abandon an I/O producer.
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        if child is not None:
            report.update(native_run_executed=True, exit_code=child.returncode)
        raise
    finally:
        report.update(native_elapsed_seconds=time.perf_counter()-started,
                      child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    decoder = validate_prefix if prefix else validate_formation
    options = dict(coupled_extension=coupled_extension)
    if realization is not None:
        options['source_realization_config'] = source_realization_config
    decoder(runtime, runtime/'observations', **options)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--coupled-extension', action='store_true',
                        help='Execute only a verified168106-unit computational extension build')
    parser.add_argument('--scope', choices=FORMATION_SCOPES, default='full-formation')
    parser.add_argument('--source-realization-config', type=Path)
    args = parser.parse_args()
    result = execute(args.runtime, coupled_extension=args.coupled_extension, scope=args.scope,
                     source_realization_config=args.source_realization_config)
    print(json.dumps({k: result[k] for k in ('status', 'complete_formation_executed', 'complete_formation_retrieval')}))


if __name__ == '__main__':
    main()
