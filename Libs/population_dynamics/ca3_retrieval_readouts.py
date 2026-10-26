"""Prospectively specified, descriptive CA3 source retrieval event readouts.

The three cues and their unequal138/137 memberships come from the pinned source,
not retrieval outcomes. Event timestamps do not independently verify delivered
currents or the terminal native clock. No retrieval-success threshold is applied.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .carlsim_records import SOURCE_REVISION, network_header, spike_blocks, validate_coupled_identity
from .provenance import sha256_file

AUTHOR_REVISION = '1989764942516bf110dafc29ad08d815fd699f4c'
MAIN_PATH = 'src/main_ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275.cpp'
AUTHOR_MAIN_SHA256 = 'b45dfc11375495b2f49f5b38d7104a29e11323d29b1d66330f88b03ac72a458a'
OVERLAY_SHA256 = '84e65574399bfcd71b48f74d96bb3d66cf3bb5c6e389e7eab3d3dd1de7703f4a'
CONFIG_SHA256 = '8bb51499780570af1cb499d5f61442408439000605a940e200407da6447a53ad'
SETUP_SHA256 = 'b6dc5216827055124226f907795857c81e4997ee0b9cf49ce3422d54d701c3ef'
PATTERN_SHA256 = '9f9c6376509ccb01b4acc7d1e47feb81dd308658e29186304feb616c72330f91'
INPUT_NETWORK = 'network65p_assem275_divnorm.dat'
PYRAMIDAL = 'CA3_Pyramidal'
POPULATIONS = {PYRAMIDAL:75377, 'CA3_Axo_Axonic':1482, 'CA3_BC_CCK':168,
               'CA3_Basket':115, 'CA3_Bistratified':2281, 'CA3_Ivy':1459,
               'CA3_MFA_ORDEN':562, 'CA3_QuadD_LM':2609}
CORE_UNITS = 84053
DURATION_MS = 1001
WINDOW_START_MS = np.array([[500,520],[730,750],[960,980]], dtype=np.int64)
WINDOW_STOP_MS = np.array([[520,540],[750,770],[980,1000]], dtype=np.int64)
ASSEMBLY_STARTS = (0,275,550)
ASSEMBLY_UNITS = 275
ELIGIBLE_UNITS = 138
WINDOW_SECONDS = .020


# Exact outputs of the declared twofold transformation of the pinned author
# files, including the local550 checkpoint binding. Caller tests regenerate
# these bytes from the original source, so altered protocols cannot pass by
# copying a self-declared staged hash into a receipt.
COUPLED_OVERLAY_SHA256 = 'cf88ebeff742ce83893a8ced810bd0aca1bda5cfc058c125b88251b06ab51bfb'
COUPLED_CONFIG_SHA256 = '1852603fdebe0bf38c4b343063dba99fadc8ac9e82a545f26833e685ffe6acba'
COUPLED_PATTERN_SHA256 = 'bd7ac79a578d8de21eac79dea979a4280a5726a30f790f0907c15fe5860cfe33'


def source_identity(*, coupled_extension=False, neuron_substeps=None):
    """Select the explicitly requested source family; never infer it from data."""
    scale = 2 if coupled_extension else 1
    overlay = COUPLED_OVERLAY_SHA256 if coupled_extension else OVERLAY_SHA256
    if neuron_substeps is not None:
        from .ca3_neuron_substeps import MAIN_SHA256, validate_count

        validate_count(neuron_substeps)
        if not coupled_extension:
            raise ValueError('CA3_REFINEMENT_REQUIRES_COUPLED_RETRIEVAL')
        overlay = MAIN_SHA256[neuron_substeps]
    return dict(core_units=CORE_UNITS*scale,
        populations={name: count*scale for name, count in POPULATIONS.items()},
        assembly_units=ASSEMBLY_UNITS*scale, eligible_units=ELIGIBLE_UNITS*scale,
        assembly_starts=tuple(start*scale for start in ASSEMBLY_STARTS),
        pattern_integer_count=1100*scale, entries_per_phase=55*scale,
        input_network='network65p_assem550_divnorm.dat' if coupled_extension else INPUT_NETWORK,
        overlay_sha256=overlay,
        config_sha256=COUPLED_CONFIG_SHA256 if coupled_extension else CONFIG_SHA256,
        pattern_sha256=COUPLED_PATTERN_SHA256 if coupled_extension else PATTERN_SHA256)


def source_masks(units, *, coupled_extension=False):
    """Memberships use source group-local IDs; no random halves or outcome selection."""
    identity = source_identity(coupled_extension=coupled_extension)
    assembly, eligible = identity['assembly_units'], identity['eligible_units']
    starts = identity['assembly_starts']
    if type(units) is not int or units < 3*assembly:
        raise ValueError('CA3_THREE_COMPLETE_SOURCE_ASSEMBLIES_REQUIRED')
    ids = np.arange(units)
    target = np.stack([(ids >= a+eligible) & (ids < a+assembly) for a in starts])
    eligible = np.stack([(ids >= a) & (ids < a+eligible) for a in starts])
    other = target.any(axis=0)[None, :] & ~target
    return dict(target_never_boosted=target, other_never_boosted=other, eligible_for_boost=eligible)


def _count_records(blocks, units, *, windows):
    full = np.zeros(units,dtype=np.int64)
    per_ms = np.zeros(DURATION_MS,dtype=np.int64)
    counted = np.zeros((3,2,units),dtype=np.int64) if windows else None
    records, first, last = 0, None, None
    for block in blocks:
        rows = np.asarray(block)
        if (rows.ndim != 2 or rows.shape[1] != 2 or rows.dtype.kind != 'i'
                or np.any(rows < 0) or np.any(rows[:,0] >= DURATION_MS) or np.any(rows[:,1] >= units)):
            raise ValueError('CA3_RAW_EVENT_ID_OR_TIME_INVALID')
        if not len(rows):
            continue
        # add.at keeps repeated event records; no chronological ordering is assumed.
        np.add.at(full,rows[:,1],1)
        np.add.at(per_ms,rows[:,0],1)
        records += len(rows)
        first = int(rows[:,0].min()) if first is None else min(first,int(rows[:,0].min()))
        last = int(rows[:,0].max()) if last is None else max(last,int(rows[:,0].max()))
        if windows:
            for cue in range(3):
                for phase in range(2):
                    selected = (rows[:,0] >= WINDOW_START_MS[cue,phase]) & (rows[:,0] < WINDOW_STOP_MS[cue,phase])
                    np.add.at(counted[cue,phase],rows[selected,1],1)
    return dict(full_counts=full,per_ms=per_ms,window_counts=counted,records=records,
                first_event_ms=first,last_event_ms=last)


def count_pyramidal_windows(blocks, units, *, coupled_extension=False):
    source_masks(units, coupled_extension=coupled_extension)
    return _count_records(blocks,units,windows=True)


def _read(path):
    return json.loads(Path(path).read_text())


def _source_proof(record, *, coupled_extension=False, neuron_substeps=None,
                  source_realization_config=None, runtime=None):
    if record.get('retrieval_condition', 'source') != 'source':
        raise ValueError('CA3_SOURCE_READOUT_REQUIRES_SOURCE_CONDITION')
    if (neuron_substeps is not None or 'numerical_refinement' in record
            or record.get('integration', 'RUNGE_KUTTA4_5') != 'RUNGE_KUTTA4_5'):
        from .ca3_neuron_substeps import validate_report

        validate_report(record, count=neuron_substeps)
    identity = source_identity(coupled_extension=coupled_extension, neuron_substeps=neuron_substeps)
    if source_realization_config is not None or 'source_realization' in record:
        from .ca3_realization import bind_runtime

        if source_realization_config is None:
            raise ValueError('CA3_REALIZATION_EXPLICIT_CONFIGURATION_REQUIRED')
        if runtime is None or not coupled_extension or neuron_substeps is not None:
            raise ValueError('CA3_REALIZATION_COUPLED_SOURCE_RUNTIME_REQUIRED')
        bind_runtime(Path(runtime), record, source_realization_config)
        identity['overlay_sha256'] = record['overlay_source_sha256']
    entries = record['source_file_byte_checks']
    indexed = {row['path']:row for row in entries}
    if len(indexed) != len(entries):
        raise ValueError('CA3_SOURCE_PROOF_DUPLICATE')
    expected = {
        MAIN_PATH:(False,not coupled_extension,AUTHOR_MAIN_SHA256,identity['overlay_sha256']),
        'generateCONFIGStateSTP.h':(not coupled_extension,False,CONFIG_SHA256,identity['config_sha256']),
        'generateSETUPStateSTP.h':(True,False,SETUP_SHA256,SETUP_SHA256),
    }
    for name,values in expected.items():
        row = indexed.get(name,{})
        actual = tuple(row.get(k) for k in ('source_bytes_preserved','source_path_binding_only','source_sha256','staged_sha256'))
        if actual != values or (coupled_extension and row.get('coupled_extension_transformation_verified') is not True):
            raise ValueError('CA3_BOUND_SOURCE_PROGRAM_REQUIRED:'+name)
    return identity


def _bound_inputs(runtime, *, coupled_extension=False, neuron_substeps=None, source_realization_config=None):
    identity = source_identity(coupled_extension=coupled_extension, neuron_substeps=neuron_substeps)
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    relative = ('build_report.json','native_report.json','observations/recording.json',
                'observations/population_counts.npz','manifest.json','bin/ca3_retrieval',
                'trainPat.csv','native_run/trainPat.csv')
    hashes = {name:sha256_file(runtime/name) for name in relative}
    build, native, recording = (_read(runtime/name) for name in relative[:3])
    proofs = [_source_proof(record, coupled_extension=coupled_extension,
                           neuron_substeps=neuron_substeps,
                           source_realization_config=source_realization_config, runtime=runtime)
              for record in (build, native)]
    if proofs[0] != proofs[1]:
        raise ValueError('CA3_BUILD_NATIVE_SOURCE_PROOFS_DIFFER')
    identity = proofs[0]
    if source_realization_config is not None:
        hashes['realization_source.cpp'] = sha256_file(runtime/'realization_source.cpp')
        hashes['source_realization.yaml'] = sha256_file(runtime/'source_realization.yaml')
        if (hashes['source_realization.yaml'] != sha256_file(source_realization_config)
                or recording.get('source_realization') != native['source_realization']
                or recording.get('realization_circuit_binding') != native.get('realization_circuit_binding')):
            raise ValueError('CA3_REALIZATION_NATIVE_DECODER_IDENTITY_DIFFERS')
    expected = dict(scope='coupled_extension_retrieval' if coupled_extension else 'native_source_retrieval',expected_duration_ms=DURATION_MS,
                    framework_revision=SOURCE_REVISION,author_revision=AUTHOR_REVISION,
                    core_neurons=identity['core_units'],populations=identity['populations'],pattern_sha256=identity['pattern_sha256'],
                    pattern_integer_count=identity['pattern_integer_count'],overlay_source_sha256=identity['overlay_sha256'],
                    original_author_main_sha256=AUTHOR_MAIN_SHA256,
                    retrieval_program_changed_only_in_input_path=not coupled_extension)
    for record in (build,native):
        if coupled_extension:
            validate_coupled_identity(record, neuron_substeps=neuron_substeps)
            extension = record['coupled_extension']
            if (extension.get('source_pattern_sha256') != PATTERN_SHA256
                    or extension.get('derived_pattern_sha256') != identity['pattern_sha256']):
                raise ValueError('CA3_COUPLED_PATTERN_PROVENANCE_CHANGED')
        if any(record.get(key) != value for key,value in expected.items()):
            raise ValueError('CA3_PINNED_COMPLETE_RETRIEVAL_SOURCE_REQUIRED')
        contract = record['retrieval_input_contract']
        required = dict(effective_input_binding=identity['input_network'],source_path_binding_only=not coupled_extension,
                        source_carlsim_seed=10,source_host_srand_seed=3,source_estdp_enabled=True)
        if coupled_extension:
            required['fresh_coupled_checkpoint_required'] = True
        if any(contract.get(key) != value for key,value in required.items()):
            raise ValueError('CA3_SOURCE_CUE_AND_PLASTICITY_CONTRACT_CHANGED')
    if (neuron_substeps is not None
            and recording.get('numerical_refinement') != native['numerical_refinement']):
        raise ValueError('CA3_DECODED_REFINEMENT_IDENTITY_DIFFERS')
    if coupled_extension and (build['coupled_extension'] != native['coupled_extension']
            or build['coupled_extension_config'] != native['coupled_extension_config']
            or recording.get('coupled_extension') != native['coupled_extension']):
        raise ValueError('CA3_COUPLED_BUILD_NATIVE_DECODER_IDENTITY_DIFFERS')
    expected_build = ('CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING'
                      if source_realization_config is not None else
                      'COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED' if coupled_extension
                      else 'NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING')
    if (build['status'] != expected_build
            or native['status'] != family+'_RETRIEVAL_EXIT_ZERO' or native['exit_code'] != 0
            or native.get('native_run_executed') is not True or native.get('retrieval_executed') is not True
            or native.get('normalized_input_qualified') is not True
            or build['binary_sha256'] != hashes['bin/ca3_retrieval']
            or native['binary_sha256'] != build['binary_sha256']
            or native['runtime_manifest_sha256'] != hashes['manifest.json']):
        raise ValueError('CA3_COMPLETED_BOUND_NATIVE_RETRIEVAL_REQUIRED')
    if (recording['status'] != 'DECODED_'+family+'_RETRIEVAL_FILES' or recording['duration_ms'] != DURATION_MS
            or recording['time_resolution_ms'] != 1 or recording['units'] != identity['core_units']
            or recording.get('retrieval_executed') is not True
            or recording['native_report_sha256'] != hashes['native_report.json']
            or recording['observation_sha256'] != hashes['observations/population_counts.npz']
            or recording['implementation_sha256'] != sha256_file(Path(__file__).with_name('carlsim_records.py'))):
        raise ValueError('CA3_BOUND_DECODED_1001MS_RECORDING_REQUIRED')
    if hashes['trainPat.csv'] != identity['pattern_sha256'] or hashes['native_run/trainPat.csv'] != identity['pattern_sha256']:
        raise ValueError('CA3_TRAIN_PATTERN_CHANGED')
    pattern = np.asarray([int(token) for token in (runtime/'trainPat.csv').read_text().split()],dtype=np.int64)
    if pattern.shape != (identity['pattern_integer_count'],) or np.any(pattern < 0) or np.any(pattern >= identity['assembly_units']):
        raise ValueError('CA3_SOURCE_PATTERN_INDICES_INVALID')
    # Header only: the previous native/decode receipts bind the normalized
    # synaptic input; event-only analysis does not rehash its large payload.
    header = network_header(runtime/'native_run'/identity['input_network'])
    groups = header['groups']
    declared = [dict(asdict(group),grid=list(group.grid)) for group in groups]
    if (header['units'] != identity['core_units'] or {g.name:g.units for g in groups} != identity['populations']
            or header['simulation_seconds'] != 45.849998474121094 or recording['groups'] != declared):
        raise ValueError('CA3_RECORDING_GROUP_IDENTITIES_CHANGED')
    with np.load(runtime/'observations/population_counts.npz',allow_pickle=False) as saved:
        if set(saved.files) != {'unit_ids','unit_counts','population_counts_per_ms','group_names'}:
            raise ValueError('CA3_SAVED_COUNT_SCHEMA_CHANGED')
        counts = {key:saved[key] for key in saved.files}
    if (counts['unit_ids'].dtype.kind != 'i' or not np.array_equal(counts['unit_ids'],np.arange(identity['core_units']))
            or counts['unit_counts'].shape != (identity['core_units'],) or counts['unit_counts'].dtype != np.dtype('int64')
            or counts['population_counts_per_ms'].shape != (DURATION_MS,len(groups))
            or counts['population_counts_per_ms'].dtype != np.dtype('int64')
            or np.any(counts['unit_counts'] < 0) or np.any(counts['population_counts_per_ms'] < 0)
            or not np.array_equal(counts['group_names'],np.asarray([g.name for g in groups]))):
        raise ValueError('CA3_SAVED_COUNT_IDENTITIES_CHANGED')
    return build,native,recording,groups,counts,pattern,hashes


def _subset_summary(baseline, cue, mask):
    units = int(mask.sum())
    base, stimulated = baseline[mask],cue[mask]
    base_count, cue_count = int(base.sum()),int(stimulated.sum())
    base_active, cue_active = int(np.count_nonzero(base)),int(np.count_nonzero(stimulated))
    return dict(units=units,baseline_event_count=base_count,cue_event_count=cue_count,
                baseline_rate_hz=base_count/(units*WINDOW_SECONDS),cue_rate_hz=cue_count/(units*WINDOW_SECONDS),
                baseline_active_units=base_active,cue_active_units=cue_active,
                baseline_spiking_fraction=base_active/units,cue_spiking_fraction=cue_active/units,
                delta_rate_hz=(cue_count-base_count)/(units*WINDOW_SECONDS),
                delta_spiking_fraction=(cue_active-base_active)/units)


def analyze(runtime, output, *, coupled_extension=False, neuron_substeps=None, source_realization_config=None):
    identity = source_identity(coupled_extension=coupled_extension, neuron_substeps=neuron_substeps)
    runtime, output = Path(runtime),Path(output)
    if output.exists():
        raise FileExistsError(output)
    build,native,recording,groups,saved,pattern,hashes = _bound_inputs(
        runtime, coupled_extension=coupled_extension, neuron_substeps=neuron_substeps,
        source_realization_config=source_realization_config)
    if source_realization_config is not None:
        identity["overlay_sha256"] = build["overlay_source_sha256"]
    files = {row['group']:row for row in recording['files']}
    if len(files) != len(recording['files']) or set(files) != {g.name for g in groups}:
        raise ValueError('CA3_COMPLETE_UNIQUE_EVENT_FILES_REQUIRED')
    total_records = 0
    checked = []
    pyramidal = None
    for group in groups:
        row = files[group.name]
        relative = 'native_run/results/spk_'+group.name+'.dat'
        path = runtime/relative
        if row['relative_path'] != relative or sha256_file(path) != row['sha256']:
            raise ValueError('CA3_RAW_EVENT_FILE_CHANGED:'+group.name)
        blocks = spike_blocks(path,units=group.units,stop_ms=DURATION_MS)
        current = (count_pyramidal_windows(blocks,group.units,coupled_extension=coupled_extension) if group.name == PYRAMIDAL
                   else _count_records(blocks,group.units,windows=False))
        if (not np.array_equal(current['full_counts'],saved['unit_counts'][group.start:group.stop])
                or not np.array_equal(current['per_ms'],saved['population_counts_per_ms'][:,group.group_id])):
            raise ValueError('CA3_RAW_AND_SAVED_COUNTS_DIFFER:'+group.name)
        if any(current[key] != row[key] for key in ('records','first_event_ms','last_event_ms')):
            raise ValueError('CA3_RECORDED_EVENT_EXTENT_OR_MULTIPLICITY_CHANGED:'+group.name)
        if sha256_file(path) != row['sha256']:
            raise ValueError('CA3_RAW_EVENT_FILE_CHANGED_DURING_READ:'+group.name)
        total_records += current['records']
        checked.append(dict(group=group.name,relative_path=relative,sha256=row['sha256'],
                            units=group.units,records=current['records'],full_record_counts_exact=True))
        if group.name == PYRAMIDAL:
            pyramidal,group_pyr = current,group
    if (pyramidal is None or total_records != recording['records']
            or total_records != int(saved['unit_counts'].sum())
            or total_records != int(saved['population_counts_per_ms'].sum())):
        raise ValueError('CA3_COMPLETE_EVENT_DENOMINATOR_CHANGED')
    masks = source_masks(group_pyr.units,coupled_extension=coupled_extension)
    rows = []
    for cue in range(3):
        row = dict(cue_id=cue+1,assembly_local_start=identity['assembly_starts'][cue],
                   assembly_global_start=group_pyr.start+identity['assembly_starts'][cue],
                   baseline_start_ms=int(WINDOW_START_MS[cue,0]),baseline_stop_ms=int(WINDOW_STOP_MS[cue,0]),
                   cue_start_ms=int(WINDOW_START_MS[cue,1]),cue_stop_ms=int(WINDOW_STOP_MS[cue,1]),
                   following_source_background_ms=(210,210,1)[cue])
        baseline,stimulated = pyramidal['window_counts'][cue]
        for prefix,key in (('target','target_never_boosted'),('other','other_never_boosted'),('eligible','eligible_for_boost')):
            row.update({prefix+'_'+stat:value for stat,value in _subset_summary(baseline,stimulated,masks[key][cue]).items()})
        for measure in ('rate_hz','spiking_fraction'):
            for phase in ('baseline','cue','delta'):
                key=phase+'_'+measure
                row['target_minus_other_'+key]=row['target_'+key]-row['other_'+key]
        rows.append(row)
    for name,expected in hashes.items():
        if sha256_file(runtime/name) != expected:
            raise ValueError('CA3_READOUT_INPUT_CHANGED_DURING_ANALYSIS:'+name)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'pyramidal_window_counts.npz',
        pyramidal_global_unit_ids=np.arange(group_pyr.start,group_pyr.stop,dtype=np.int64),
        pyramidal_local_unit_ids=np.arange(group_pyr.units,dtype=np.int64),cue_ids=np.arange(1,4,dtype=np.int64),
        window_names=np.asarray(['baseline','cue']),window_start_ms=WINDOW_START_MS,window_stop_ms=WINDOW_STOP_MS,
        window_counts=pyramidal['window_counts'],full_record_unit_counts=pyramidal['full_counts'],
        scheduled_train_pattern_indices=pattern,**{key+'_mask':value for key,value in masks.items()})
    with (output/'descriptive_readouts.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report = dict(status=('DESCRIBED_COUPLED_EXTENSION_CA3_RETRIEVAL_EVENTS' if coupled_extension
                         else 'DESCRIBED_SOURCE_SCHEDULED_CA3_RETRIEVAL_EVENTS'),
        created_utc=datetime.now(timezone.utc).isoformat(),runtime=str(runtime.resolve()),
        duration_ms=DURATION_MS,time_resolution_ms=1,cues_retained=3,
        source_scheduled_window_start_ms=WINDOW_START_MS.tolist(),source_scheduled_window_stop_ms=WINDOW_STOP_MS.tolist(),
        raw_records=total_records,pyramidal_records=pyramidal['records'],
        full_record_recounted_units=identity['core_units'],window_counted_pyramidal_units=group_pyr.units,
        target_never_boosted_units=identity['assembly_units']-identity['eligible_units'],
        other_never_boosted_units=2*(identity['assembly_units']-identity['eligible_units']),eligible_for_boost_units=identity['eligible_units'],
        source_subset_identity=f"group-local contiguous{identity['assembly_units']}-cell assemblies; first{identity['eligible_units']} eligible, remaining{identity['assembly_units']-identity['eligible_units']} never boosted by source code",
        unit_identity='global IDs from normalized-network group header; group-local indices retained explicitly',
        count_semantics='all file records including duplicates; file order may be nonchronological; half-open integer-ms windows',
        pattern_sha256=identity['pattern_sha256'],pattern_integer_count=identity['pattern_integer_count'],source_pattern_phases=20,source_entries_per_phase=identity['entries_per_phase'],
        source_estdp_enabled=True,source_random_shuffle_active=False,
        overlay_source_sha256=identity['overlay_sha256'],original_author_main_sha256=AUTHOR_MAIN_SHA256,
        source_config_sha256=identity['config_sha256'],source_setup_sha256=SETUP_SHA256,
        input_sha256=hashes,checked_event_files=checked,
        native_report_sha256=hashes['native_report.json'],recording_sha256=hashes['observations/recording.json'],
        normalized_input_sha256=native['normalized_input_sha256'],normalized_network_payload_rehashed=False,
        full_record_unit_counts_match=True,full_record_population_counts_per_ms_match=True,
        actual_current_trace_verified=False,terminal_native_clock_independently_verified=False,
        independent_native_spike_counter_verified=False,input_network_is_microstate_restart=False,
        third_cue_post_source_support_ms=1,matched_long_post_cue_analysis_performed=False,
        scientific_fidelity_qualified=False,pattern_completion_qualified=False,causal_claim_qualified=False,
        neuroscientific_claim_qualified=False,full_scale_qualified=False,production_episodes=0,
        inferential_statistics_performed=False,p_values=False,confidence_intervals=False,success_thresholds_applied=False,
        interpretation='descriptive source-scheduled activity and recruitment contrasts; no source retrieval-success threshold',
        limitations=['All Pyramidal background-current distributions change during cues; inhibitory currents are held for20ms.',
                     'Source eligibility does not establish actual current delivery; event bounds do not independently verify internal clock.',
                     'Other assemblies have different previous-cue histories; neurons and the three cues are not independent biological replicates.',
                     'Third cue has only1ms of source-scheduled post support; no matched long persistence comparison.'],
        implementation_sha256=sha256_file(Path(__file__)),
        artifacts={name:sha256_file(output/name) for name in ('pyramidal_window_counts.npz','descriptive_readouts.csv')})
    if coupled_extension:
        report.update(coupled_extension=native['coupled_extension'],
                      coupled_extension_config=native['coupled_extension_config'])
    if neuron_substeps is not None:
        report.update(numerical_refinement=native['numerical_refinement'], integration=native['integration'],
                      interpretation='descriptive coupled retrieval under explicit neuronal integration refinement')
    if source_realization_config is not None:
        report['source_realization'] = native['source_realization']
        report['realization_circuit_binding'] = native['realization_circuit_binding']
    with (output/'report.json').open('x') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coupled-extension',action='store_true')
    parser.add_argument('--neuron-substeps', type=int, choices=(5, 10, 20))
    parser.add_argument('--source-realization-config', type=Path)
    args=parser.parse_args()
    print(json.dumps(analyze(args.runtime,args.output,coupled_extension=args.coupled_extension,
                            neuron_substeps=args.neuron_substeps, source_realization_config=args.source_realization_config),indent=2))


if __name__ == '__main__':
    main()
