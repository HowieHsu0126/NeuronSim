"""Read the pinned CARLsim CA3 source's network header and AER event files.

Writer sources: carlsim/kernel/src/snn_manager.cpp:1867-2050,8404-8515;
carlsim/monitor/spike_monitor_core.cpp:73-74,426-453 at revision
c4955998b950b9c561b3373a71b611aedc705be9. The nominal grid is an internal
index layout, not anatomical coordinates. saveSimulation is not a full-state
restart: its payload does not include neuronal, event-queue or RNG state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterator

import numpy as np

SPIKE_SIGNATURE = 206661989
NETWORK_SIGNATURE = 294338571
SPIKE_HEADER = struct.Struct('<ifiii')
NETWORK_HEADER = struct.Struct('<ifffii')
GROUP_HEADER = struct.Struct('<iiiii100s')
FLOAT32_SPIKE_VERSION = struct.unpack('<f', struct.pack('<f', 0.2))[0]
FLOAT32_NETWORK_VERSION = struct.unpack('<f', struct.pack('<f', 0.3))[0]
SOURCE_REVISION = 'c4955998b950b9c561b3373a71b611aedc705be9'

CA3_REFERENCE_POPULATIONS = {
    "CA3_QuadD_LM": 2609, "CA3_Axo_Axonic": 1482, "CA3_Basket": 115,
    "CA3_BC_CCK": 168, "CA3_Bistratified": 2281, "CA3_Ivy": 1459,
    "CA3_MFA_ORDEN": 562, "CA3_Pyramidal": 75377,
}


def validate_coupled_identity(report: dict, *, neuron_substeps: int | None = None) -> None:
    """Bind the explicit twofold computational model, never native-source claims."""
    if (neuron_substeps is not None or 'numerical_refinement' in report
            or report.get('integration', 'RUNGE_KUTTA4_5') != 'RUNGE_KUTTA4_5'):
        from .ca3_neuron_substeps import validate_report

        validate_report(report, count=neuron_substeps)
    scope = report.get('scope')
    if scope not in ('coupled_extension_prefix', 'coupled_extension_formation',
                     'coupled_extension_retrieval'):
        raise ValueError('CA3_COUPLED_SCOPE_REQUIRED')
    role = 'retrieval' if scope.endswith('retrieval') else 'formation'
    populations = {name: count * 2 for name, count in CA3_REFERENCE_POPULATIONS.items()}
    extension, config = report.get('coupled_extension'), report.get('coupled_extension_config')
    if (not isinstance(extension, dict) or not isinstance(config, dict)
            or config.get('resolved') != {'population_multiplier': 2}
            or type(config['resolved'].get('population_multiplier')) is not int):
        raise ValueError('CA3_COUPLED_DECLARATION_REQUIRED')
    digest = config.get('sha256')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('CA3_COUPLED_CONFIG_IDENTITY_REQUIRED')
    expected = dict(schema='ca3_coupled_extension_v1',
                    status='PROSPECTIVE_UNQUALIFIED_COUPLED_EXTENSION',
                    multiplier=2, source_role=role, source_core_neurons=84053,
                    derived_core_neurons=168106,
                    source_populations=CA3_REFERENCE_POPULATIONS,
                    derived_populations=populations, source_assembly_size=275,
                    derived_assembly_size=550,
                    source_cue_size=138 if role == 'retrieval' else 275,
                    derived_cue_size=276 if role == 'retrieval' else 550)
    if (any(extension.get(key) != value for key, value in expected.items())
            or type(extension.get('multiplier')) is not int
            or report.get('core_neurons') != 168106
            or type(report.get('core_neurons')) is not int
            or report.get('populations') != populations
            or any(type(value) is not int for value in report['populations'].values())
            or report.get('source_code_semantics_preserved_before_stop') is not False
            or extension.get('fresh_native_graph_required') is not True
            or extension.get('fresh_formation_and_normalized_checkpoint_required') is not True
            or extension.get('graph_or_output_cloning') is not False):
        raise ValueError('CA3_COUPLED_MODEL_IDENTITY_DIFFERS')


@dataclass(frozen=True)
class Group:
    group_id: int
    name: str
    start: int
    stop: int  # exclusive global unit boundary
    grid: tuple[int, int, int]

    @property
    def units(self):
        return self.stop-self.start


def _read_exact(stream, size):
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError('CARLSIM_TRUNCATED_HEADER')
    return payload


def network_header(path: Path):
    """Parse metadata only; do not claim validation of the synaptic payload."""
    with Path(path).open('rb') as stream:
        signature, version, seconds, execution_seconds, units, group_count = NETWORK_HEADER.unpack(
            _read_exact(stream, NETWORK_HEADER.size))
        if signature != NETWORK_SIGNATURE or version != FLOAT32_NETWORK_VERSION:
            raise ValueError('CARLSIM_NETWORK_FORMAT_UNSUPPORTED')
        if (units <= 0 or not 0 < group_count <= units or not math.isfinite(seconds)
                or not math.isfinite(execution_seconds) or min(seconds, execution_seconds) < 0):
            raise ValueError('CARLSIM_NETWORK_METADATA_INVALID')
        groups = []
        for index in range(group_count):
            start, end, nx, ny, nz, name = GROUP_HEADER.unpack(_read_exact(stream, GROUP_HEADER.size))
            name = name.split(b'\0', 1)[0].decode('utf-8')
            if (not name or '/' in name or '\\' in name or name in ('.', '..')
                    or not 0 <= start <= end < units or min(nx, ny, nz) <= 0
                    or nx*ny*nz != end-start+1):
                raise ValueError('CARLSIM_GROUP_IDENTITY_INVALID')
            groups.append(Group(index, name, start, end+1, (nx, ny, nz)))
        if len({g.name for g in groups}) != group_count:
            raise ValueError('CARLSIM_GROUP_NAMES_DUPLICATE')
        cursor = 0
        for group in sorted(groups, key=lambda g:g.start):
            if group.start != cursor:
                raise ValueError('CARLSIM_GLOBAL_UNIT_COVERAGE_INVALID')
            cursor = group.stop
        if cursor != units:
            raise ValueError('CARLSIM_GLOBAL_UNIT_COVERAGE_INVALID')
        return dict(units=units, groups=groups, simulation_seconds=seconds,
                    recorded_execution_seconds=execution_seconds, payload_offset=stream.tell(),
                    full_state_restart_available=False)


def spike_blocks(path: Path, *, units: int, stop_ms: int,
                 chunk_records: int = 65536) -> Iterator[np.ndarray]:
    """Yield unmodified (integer-ms time, group-local ID) records in file order.

The source writes D2 and D1 tables separately, so global time order is not an
invariant. Duplicates are retained. The native monitor writes [0, stop_ms).
"""
    if type(units) is not int or units <= 0 or type(stop_ms) is not int or stop_ms <= 0:
        raise ValueError('CARLSIM_EVENT_BOUNDS_INVALID')
    if type(chunk_records) is not int or chunk_records <= 0:
        raise ValueError('CARLSIM_EVENT_CHUNK_INVALID')
    with Path(path).open('rb') as stream:
        signature, version, nx, ny, nz = SPIKE_HEADER.unpack(_read_exact(stream, SPIKE_HEADER.size))
        if signature != SPIKE_SIGNATURE or version != FLOAT32_SPIKE_VERSION:
            raise ValueError('CARLSIM_SPIKE_FORMAT_UNSUPPORTED')
        if min(nx, ny, nz) <= 0 or nx*ny*nz != units:
            raise ValueError('CARLSIM_SPIKE_UNIT_COUNT_MISMATCH')
        while True:
            payload = stream.read(chunk_records*8)
            if not payload:
                break
            if len(payload) % 8:
                raise ValueError('CARLSIM_TRUNCATED_EVENT_RECORD')
            rows = np.frombuffer(payload, dtype='<i4').reshape(-1, 2)
            if ((rows < 0).any() or (rows[:,0] >= stop_ms).any()
                    or (rows[:,1] >= units).any()):
                raise ValueError('CARLSIM_EVENT_TIME_OR_ID_OUT_OF_RANGE')
            yield rows


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_native(runtime: Path, output: Path, *, scope: str, coupled_extension: bool = False,
                     cue_omission: bool = False, neuron_substeps: int | None = None,
                     source_realization_config: Path | None = None):
    """Decode a completed native source run without promoting it to production."""
    runtime, output = Path(runtime), Path(output)
    report_path = runtime/'native_report.json'
    report = json.loads(report_path.read_text())
    realization = None
    if source_realization_config is not None or 'source_realization' in report:
        from .ca3_realization import bind_runtime

        realization = bind_runtime(runtime, report, source_realization_config)
        if not coupled_extension or cue_omission or neuron_substeps is not None:
            raise ValueError('CA3_REALIZATION_SOURCE_RETRIEVAL_DECODER_REQUIRED')
    if neuron_substeps is not None and (scope != 'retrieval' or not coupled_extension or cue_omission):
        raise ValueError('CA3_REFINEMENT_REQUIRES_COUPLED_SOURCE_RETRIEVAL')
    formation = scope == 'full-formation'
    retrieval = scope == 'retrieval'
    condition = 'cue_omission' if cue_omission else 'source'
    if type(cue_omission) is not bool or (cue_omission and not retrieval):
        raise ValueError('CA3_OMISSION_RETRIEVAL_SCOPE_REQUIRED')
    if retrieval and report.get('retrieval_condition', 'source') != condition:
        raise ValueError('CA3_RETRIEVAL_CONDITION_FLAGS_DIFFER')
    if cue_omission:
        from .ca3_cue_omission import validate_report

        validate_report(report, population_multiplier=2 if coupled_extension else 1)
    kind = 'RETRIEVAL' if retrieval else 'FORMATION' if formation else 'PREFIX'
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    if coupled_extension:
        validate_coupled_identity(report, neuron_substeps=neuron_substeps)
    expected_status = family+'_'+kind+'_EXIT_ZERO'
    expected_scope = ('coupled_extension_' if coupled_extension else 'native_source_')+kind.lower()
    if (report['status'] != expected_status or report['exit_code'] != 0
            or report['framework_revision'] != SOURCE_REVISION
            or report['scope'] != expected_scope or not report['native_run_executed']):
        raise ValueError('CARLSIM_COMPLETED_PINNED_'+kind+'_REQUIRED')
    if formation and (report.get('expected_duration_ms') != 70000
                      or report.get('complete_formation_executed') is not True):
        raise ValueError('CARLSIM_COMPLETE_FORMATION_REQUIRED')
    if retrieval and (report.get('expected_duration_ms') != 1001
                      or report.get('normalized_input_qualified') is not True):
        raise ValueError('CARLSIM_QUALIFIED_SOURCE_RETRIEVAL_REQUIRED')
    if output.exists():
        raise FileExistsError(output)
    run = runtime/'native_run'
    input_name = 'network65p_assem550_divnorm.dat' if coupled_extension else 'network65p_assem275_divnorm.dat'
    checkpoint = run/(input_name if retrieval else 'network100p.dat' if formation else 'network5p.dat')
    if _sha(checkpoint) != report['normalized_input_sha256' if retrieval else 'checkpoint_sha256']:
        raise ValueError('CARLSIM_SAVED_NETWORK_CHANGED')
    header = network_header(checkpoint)
    duration = report['expected_duration_ms'] if formation or retrieval else report['expected_prefix_ms']
    if type(duration) is not int or duration <= 0:
        raise ValueError('CARLSIM_PREFIX_DURATION_INVALID')
    # Retrieval loads a learned network into a new simulation; its input header
    # records the formation time, not the retrieval clock or a microscopic restart.
    checkpoint_ms = 45850 if retrieval else duration
    expected_seconds = float(np.float32(checkpoint_ms//1000)+np.float32(checkpoint_ms%1000)/np.float32(1000))
    if header['simulation_seconds'] != expected_seconds:
        raise ValueError('CARLSIM_SAVED_NETWORK_TIME_MISMATCH')
    if (header['units'] != report['core_neurons'] or
            {g.name:g.units for g in header['groups']} != report['populations']):
        raise ValueError('CARLSIM_SOURCE_POPULATION_MISMATCH')
    unit_counts = np.zeros(header['units'], dtype=np.int64)
    field_counts = np.zeros((duration, len(header['groups'])), dtype=np.int64)
    artifacts = []
    for group in header['groups']:
        path = run/'results'/('spk_'+group.name+'.dat')
        count, first, last = 0, None, None
        for rows in spike_blocks(path, units=group.units, stop_ms=duration):
            count += len(rows)
            first = int(rows[:,0].min()) if first is None else min(first, int(rows[:,0].min()))
            last = int(rows[:,0].max()) if last is None else max(last, int(rows[:,0].max()))
            unit_counts[group.start:group.stop] += np.bincount(rows[:,1], minlength=group.units)
            field_counts[:,group.group_id] += np.bincount(rows[:,0], minlength=duration)
        artifacts.append(dict(group=group.name, relative_path=str(path.relative_to(runtime)),
            sha256=_sha(path), records=count, first_event_ms=first, last_event_ms=last))
    if int(unit_counts.sum()) != int(field_counts.sum()):
        raise ValueError('CARLSIM_COUNT_PROJECTION_MISMATCH')
    output.mkdir(parents=True)
    np.savez_compressed(output/'population_counts.npz', unit_ids=np.arange(header['units']),
        unit_counts=unit_counts, population_counts_per_ms=field_counts,
        group_names=np.asarray([g.name for g in header['groups']]))
    result = dict(status='DECODED_'+family+'_'+kind+'_FILES', units=header['units'],
        model_family=family,
        duration_ms=duration, time_resolution_ms=1, records=int(unit_counts.sum()),
        groups=[asdict(g) for g in header['groups']], files=artifacts,
        native_report_sha256=_sha(report_path), implementation_sha256=_sha(Path(__file__)),
        observation_sha256=_sha(output/'population_counts.npz'),
        scope='native retrieval event-file intake; pattern-completion fidelity unverified' if retrieval else 'native formation event-file intake; retrieval unverified' if formation else 'native source prefix event-file intake; not complete formation/retrieval',
        complete_formation_executed=formation, complete_formation_retrieval=False,
        full_state_restart_available=False, synaptic_payload_validated=False,
        independent_native_spike_counter_verified=False, actual_current_trace_verified=False,
        coordinate_semantics='source cell-type index; grid is not measured anatomical geometry',
        production_episodes=0, scientific_fidelity_qualified=False, full_scale_qualified=False)
    if coupled_extension:
        result['coupled_extension'] = report['coupled_extension']
        result['scope'] = 'coupled computational extension event-file intake; scientific fidelity unverified'
    if retrieval:
        result.update(retrieval_executed=True, terminal_native_clock_independently_verified=False,
                      duration_evidence='unchanged source control flow and native exit0; all events within1001ms',
                       input_network_is_microstate_restart=False)
        if cue_omission:
            result.update(retrieval_condition='cue_omission',
                duration_evidence='source clock preserved with explicitly omitted cue overwrites; native exit0 and bounded events',
                cue_omission=report['retrieval_input_contract']['cue_omission'])
        if neuron_substeps is not None:
            result.update(numerical_refinement=report['numerical_refinement'],
                          integration=report['integration'],
                          duration_evidence='source1ms clock with explicit neuron substeps; native exit0 and bounded events')
    if realization is not None:
        result['source_realization'] = report['source_realization']
        result['realization_circuit_binding'] = report['realization_circuit_binding']
    (output/'recording.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def validate_prefix(runtime: Path, output: Path, *, coupled_extension: bool = False,
                    source_realization_config: Path | None = None):
    """Preserve the original prefix-only API and qualification boundary."""
    return _validate_native(runtime, output, scope='prefix', coupled_extension=coupled_extension,
                            source_realization_config=source_realization_config)


def validate_formation(runtime: Path, output: Path, *, coupled_extension: bool = False,
                       source_realization_config: Path | None = None):
    """Decode all units from the source70-second formation, with no retrieval claim."""
    return _validate_native(runtime, output, scope='full-formation', coupled_extension=coupled_extension,
                            source_realization_config=source_realization_config)


def validate_retrieval(runtime: Path, output: Path, *, coupled_extension: bool = False,
                        cue_omission: bool = False, neuron_substeps: int | None = None,
                      source_realization_config: Path | None = None):
    """Decode the declared retrieval condition; normalized input supplies IDs."""
    return _validate_native(runtime, output, scope='retrieval', coupled_extension=coupled_extension,
                            cue_omission=cue_omission, neuron_substeps=neuron_substeps,
                            source_realization_config=source_realization_config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('prefix', 'full-formation', 'retrieval'), default='prefix')
    parser.add_argument('--coupled-extension', action='store_true')
    parser.add_argument('--cue-omission', action='store_true')
    parser.add_argument('--neuron-substeps', type=int, choices=(5, 10, 20, 40))
    parser.add_argument('--source-realization-config', type=Path)
    args=parser.parse_args()
    validate = {'prefix':validate_prefix, 'full-formation':validate_formation, 'retrieval':validate_retrieval}[args.scope]
    if args.cue_omission and args.scope != 'retrieval':
        parser.error('--cue-omission requires --scope retrieval')
    options = dict(coupled_extension=args.coupled_extension)
    if args.cue_omission:
        options['cue_omission'] = True
    if args.neuron_substeps is not None:
        if args.scope != 'retrieval':
            parser.error('--neuron-substeps requires --scope retrieval')
        options['neuron_substeps'] = args.neuron_substeps
    if args.source_realization_config is not None:
        options['source_realization_config'] = args.source_realization_config
    print(json.dumps(validate(args.runtime,args.output,**options), indent=2))


if __name__ == '__main__':
    main()
